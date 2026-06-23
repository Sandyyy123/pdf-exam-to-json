"""
batch_runner.py — scale the parser across a whole library of PDFs.

Walks an input folder, parses every PDF, writes one JSON per file plus a manifest
that records page count, question count, figures cropped, and the share of
low-confidence questions per document. The manifest is the QA dashboard for a
library run: it tells you which files need a human or VLM second pass before the
output is trusted as production-ready.

    python batch_runner.py ./exams ./out
"""
from __future__ import annotations

import os
import sys
import json
import csv
from dataclasses import asdict

from exam_parser import parse_exam
from vlm import get_vlm


def run(in_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vlm = get_vlm()  # None unless VLM_PROVIDER is set
    manifest = []

    pdfs = [f for f in sorted(os.listdir(in_dir)) if f.lower().endswith(".pdf")]
    for fn in pdfs:
        src = os.path.join(in_dir, fn)
        stem = os.path.splitext(fn)[0]
        img_dir = os.path.join(out_dir, stem + "_figures")
        try:
            result = parse_exam(src, image_dir=img_dir, vlm_fallback=vlm)
        except Exception as e:
            manifest.append({"file": fn, "status": "ERROR", "error": str(e)})
            print(f"[ERR ] {fn}: {e}")
            continue

        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        n_q = len(result.questions)
        n_fig = sum(len(q.figures) for q in result.questions)
        low = sum(1 for q in result.questions if q.confidence < 0.6)
        manifest.append({
            "file": fn, "status": "OK", "pages": result.page_count,
            "questions": n_q, "figures": n_fig,
            "low_confidence": low,
            "low_conf_pct": round(100 * low / n_q, 1) if n_q else 0.0,
        })
        print(f"[ OK ] {fn}: {n_q} Q, {n_fig} figs, {low} low-confidence")

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        if manifest:
            w = csv.DictWriter(f, fieldnames=sorted({k for m in manifest for k in m}))
            w.writeheader()
            w.writerows(manifest)
    print(f"\nDone: {len(pdfs)} PDFs -> {out_dir} (see manifest.csv)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python batch_runner.py <in_dir> <out_dir>")
        raise SystemExit(1)
    run(sys.argv[1], sys.argv[2])
