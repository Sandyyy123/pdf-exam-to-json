> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# pdf-exam-to-json

Turn educational exam-paper PDFs into a **predefined, production-ready JSON schema** —
text, multiple-choice options, answer keys, and **per-question cropped figures** —
with deterministic parsing first and a Vision-Language-Model fallback only where
the layout needs it.

Built as a working reference for the "Parse PDFs into Structured JSON" brief:
mixed text / MCQs / math / reading comprehension / diagrams, multi-column,
rotated pages, and questions that span pages.

## Why this design

**Deterministic-first, VLM-on-demand.** PyMuPDF (`fitz`) resolves ~90% of a clean
exam paper for free and in milliseconds. A GPT-4o / Claude vision pass is invoked
**only** for questions the deterministic pass flags as low-confidence (formula-heavy
stems, fully graphical questions, scanned pages). At library scale this keeps the
per-document API cost near zero while still hitting the accuracy the brief demands.

Every question carries a `confidence` score and a `source_method` flag, so a
human or a second VLM pass can be routed to exactly the items that need it — the
output is never a black box.

## Pipeline

```
PDF ─► PyMuPDF text+image blocks ─► rotation-normalise ─► column detection
    ─► question / option / answer-key detection ─► per-question figure crop
    ─► confidence score ─► [low-confidence] ─► VLM re-extract ─► JSON
```

## Run it

```bash
pip install -r requirements.txt
python main.py                       # synthetic demo, prints structured JSON
python exam_parser.py exam.pdf out.json   # one real PDF
python batch_runner.py ./exams ./out      # whole library + QA manifest.csv
```

Enable the VLM fallback for hard layouts:

```bash
export VLM_PROVIDER=openai      # or: anthropic
python batch_runner.py ./exams ./out
```

## Files

| File | Role |
|------|------|
| `exam_parser.py` | Core engine: layout-aware extraction, column detection, rotation handling, figure cropping, confidence scoring |
| `schema.py`      | The structured JSON contract (dataclasses → `asdict` → JSON) |
| `vlm.py`         | GPT-4o / Claude vision adapters for the low-confidence fallback |
| `batch_runner.py`| Library-scale runner + per-file QA manifest |
| `main.py`        | Zero-input runnable demo |

## Edge cases handled

- **Multi-column** — left-edge x-clustering assigns a column index; reading order is column-then-vertical.
- **Rotated pages** — `page.rotation` normalised to 0 before layout reasoning.
- **Multi-page questions** — a question accumulates until the next question marker, across page breaks.
- **Answer keys** — inline (`Answer: C`) or a standalone key page, merged back by question number.
- **Figures/diagrams** — image blocks rendered to PNG at 2× and linked to the owning question with their bbox.

## Mapping to your schema

`schema.py` is the contract. Matching your predefined JSON schema is a field-mapping
config change (renames / nesting), not a rewrite — the extraction engine stays the same.
