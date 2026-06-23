"""
exam_parser.py — PDF exam paper -> structured JSON.

Core extraction engine. Reads an exam-paper PDF with PyMuPDF (fitz), detects
questions and multiple-choice options with layout-aware heuristics, crops the
figures/diagrams attached to each question, and emits one JSON object per the
schema in schema.py.

Design goals that match the job brief:
  - text + MCQ options + answer-key extraction
  - per-question image cropping (diagrams, graphs, tables-as-image)
  - multi-column handling (column detection via x-coordinate clustering)
  - rotated-page handling (page.rotation normalised before extraction)
  - multi-page questions (a question carries until the next question marker)
  - VLM fallback hook for layouts the deterministic pass can't resolve

Deterministic-first: PyMuPDF does ~90% of the work for free and fast. The VLM
(GPT-4o / Claude) is only invoked for blocks flagged low-confidence, which keeps
per-page cost near zero at library scale.
"""
from __future__ import annotations

import re
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # demo-mode fallback so the repo runs without the dep
    fitz = None

from schema import Question, ExamDocument, Option, FigureRef

# A "12." or "Q12)" or "12)" question marker at the start of a line.
QUESTION_RE = re.compile(r"^\s*(?:Q(?:uestion)?\s*)?(\d{1,3})[\.\)]\s+(.*)", re.I)
# An MCQ option: "A)", "(A)", "A.", "a)" etc.
OPTION_RE = re.compile(r"^\s*\(?([A-Ha-h])[\.\)]\s+(.*)")
# An answer-key line: "Answer: C" / "Correct answer - B" / "Ans. (D)"
ANSWER_RE = re.compile(r"(?:correct\s*answer|answer|ans)\.?\s*[:\-]?\s*\(?([A-Ha-h])\)?", re.I)
# Header that starts a standalone answer-key section ("Answer Key", "Answers").
ANSWER_KEY_HDR_RE = re.compile(r"^\s*(?:answer\s*key|answers?)\s*$", re.I)
# A row inside an answer-key section: "12. C" / "12) B" / "12 - D".
KEY_ROW_RE = re.compile(r"^\s*(\d{1,3})[\.\)\-\s]+\(?([A-Ha-h])\)?\s*$")


@dataclass
class Block:
    """A normalised text/image block with its bounding box and column index."""
    text: str
    bbox: tuple
    page: int
    column: int = 0
    is_image: bool = False
    xref: int = 0


def _detect_columns(blocks: list[Block], page_width: float) -> None:
    """Assign each block a column index by clustering left-edge x positions.

    Two-column papers cluster into two bands; single-column collapses to one.
    Mutates blocks in place.
    """
    if not blocks:
        return
    lefts = sorted(b.bbox[0] for b in blocks)
    mid = page_width / 2
    # If a meaningful share of blocks start past the midline, treat as 2-column.
    right_share = sum(1 for x in lefts if x > mid) / len(lefts)
    two_col = 0.2 < right_share < 0.8
    for b in blocks:
        b.column = 1 if (two_col and b.bbox[0] > mid) else 0


def _page_blocks(page, page_index: int) -> list[Block]:
    """Pull text + image blocks from one page in reading order, rotation-safe."""
    # Normalise rotation so coordinates are upright before we reason about layout.
    if page.rotation:
        page.set_rotation(0)
    raw = page.get_text("dict")
    pw = raw["width"]
    out: list[Block] = []
    for blk in raw["blocks"]:
        if blk.get("type") == 1:  # image block
            out.append(Block(text="", bbox=tuple(blk["bbox"]), page=page_index,
                             is_image=True, xref=blk.get("number", 0)))
            continue
        lines = []
        for ln in blk.get("lines", []):
            lines.append("".join(span["text"] for span in ln["spans"]))
        text = "\n".join(lines).strip()
        if text:
            out.append(Block(text=text, bbox=tuple(blk["bbox"]), page=page_index))
    _detect_columns(out, pw)
    # Reading order: column first, then top-to-bottom.
    out.sort(key=lambda b: (b.column, round(b.bbox[1])))
    return out


def _crop_image(page, bbox, out_dir: str, name: str, zoom: float = 2.0) -> str:
    """Render the region at `bbox` to a PNG and return its relative path."""
    os.makedirs(out_dir, exist_ok=True)
    clip = fitz.Rect(*bbox)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
    path = os.path.join(out_dir, f"{name}.png")
    pix.save(path)
    return path


def parse_exam(pdf_path: str, image_dir: str = "figures",
               vlm_fallback=None) -> ExamDocument:
    """Parse one exam PDF into an ExamDocument.

    vlm_fallback: optional callable(image_path, prompt) -> str used for blocks
    the deterministic pass flags as low-confidence (e.g. formula-heavy or
    fully graphical questions). See vlm.py for a ready GPT-4o / Claude adapter.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF not installed. `pip install pymupdf` to run for real.")

    doc = fitz.open(pdf_path)
    questions: list[Question] = []
    current: Optional[Question] = None
    answer_key: dict[int, str] = {}
    in_answer_key = False  # True once a standalone "Answer Key" section starts

    for pidx in range(len(doc)):
        page = doc[pidx]
        blocks = _page_blocks(page, pidx)

        for b in blocks:
            if b.is_image:
                # Attach figure to the question currently being built.
                if current is not None:
                    fig_name = f"q{current.number}_fig{len(current.figures)+1}"
                    path = _crop_image(page, b.bbox, image_dir, fig_name)
                    current.figures.append(FigureRef(path=path, page=pidx,
                                                     bbox=list(b.bbox)))
                continue

            for line in b.text.split("\n"):
                # A standalone "Answer Key" section flips parsing mode: from here
                # on, "N. X" rows are answer-key entries, not new questions.
                if ANSWER_KEY_HDR_RE.match(line):
                    in_answer_key = True
                    if current:
                        questions.append(current)
                        current = None
                    continue
                if in_answer_key:
                    krm = KEY_ROW_RE.match(line)
                    if krm:
                        answer_key[int(krm.group(1))] = krm.group(2).upper()
                    continue

                # Answer-key lines can appear anywhere (often a final page).
                akm = ANSWER_RE.search(line)
                qm = QUESTION_RE.match(line)
                om = OPTION_RE.match(line)

                if qm:
                    if current:
                        questions.append(current)
                    current = Question(number=int(qm.group(1)),
                                       stem=qm.group(2).strip(),
                                       page=pidx)
                elif om and current is not None:
                    current.options.append(Option(label=om.group(1).upper(),
                                                   text=om.group(2).strip()))
                elif akm and current is not None and not current.options:
                    # "Answer: C" inline within a question with no options yet
                    answer_key[current.number] = akm.group(1).upper()
                elif akm and current is not None:
                    answer_key.setdefault(current.number, akm.group(1).upper())
                elif current is not None:
                    # Continuation of the stem (multi-line / multi-page question).
                    current.stem += " " + line.strip()

    if current:
        questions.append(current)

    # Merge in any standalone answer key.
    for q in questions:
        if q.number in answer_key:
            q.correct_answer = answer_key[q.number]
        q.confidence = _score_confidence(q)
        if q.confidence < 0.6 and vlm_fallback is not None:
            _apply_vlm(q, vlm_fallback, image_dir)

    return ExamDocument(source=os.path.basename(pdf_path),
                        page_count=len(doc),
                        questions=questions)


def _score_confidence(q: Question) -> float:
    """Heuristic confidence: do we have a clean stem, options, and an answer?"""
    score = 0.4
    if len(q.stem) > 8:
        score += 0.2
    if len(q.options) >= 2:
        score += 0.3
    if q.correct_answer:
        score += 0.1
    return round(min(score, 1.0), 2)


def _apply_vlm(q: Question, vlm_fallback, image_dir: str) -> None:
    """Re-extract a low-confidence question via the VLM on its rendered crop."""
    if not q.figures:
        return
    prompt = ("Extract this exam question as JSON with keys stem, options "
              "(list of {label,text}), correct_answer. Preserve any math/LaTeX.")
    try:
        raw = vlm_fallback(q.figures[0].path, prompt)
        data = json.loads(raw)
        q.stem = data.get("stem", q.stem)
        q.options = [Option(**o) for o in data.get("options", [])] or q.options
        q.correct_answer = data.get("correct_answer", q.correct_answer)
        q.source_method = "vlm"
        q.confidence = 0.9
    except Exception:
        pass  # keep deterministic result if VLM parse fails


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python exam_parser.py <exam.pdf> [out.json]")
        raise SystemExit(1)
    result = parse_exam(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "exam.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"Wrote {out}: {len(result.questions)} questions, "
          f"{sum(len(q.figures) for q in result.questions)} figures cropped.")
