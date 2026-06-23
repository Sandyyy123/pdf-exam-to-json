"""
main.py — runnable demo.

Builds a tiny synthetic exam PDF in memory (so the repo runs with zero input
files), parses it with the real engine, and prints the structured JSON. This
exercises the actual code path: question detection, MCQ options, answer key,
multi-line stems.

    python main.py

For a real PDF:

    python exam_parser.py path/to/exam.pdf out.json
"""
import json
from dataclasses import asdict

try:
    import fitz
except ImportError:
    fitz = None

from exam_parser import parse_exam


SAMPLE = """Mathematics Exam — Section A

1. What is the value of x if 2x + 3 = 11?
A) 2
B) 4
C) 6
D) 8

2. The area of a circle with radius r is given by which formula?
A) 2 pi r
B) pi r squared
C) pi d
D) 4/3 pi r cubed

Answer Key
1. B
2. B
"""


def _make_sample_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), SAMPLE, fontsize=12)
    doc.save(path)


def main() -> None:
    if fitz is None:
        print("PyMuPDF not installed — install with: pip install -r requirements.txt")
        return
    pdf_path = "/tmp/_demo_exam.pdf"
    _make_sample_pdf(pdf_path)
    result = parse_exam(pdf_path, image_dir="/tmp/_demo_figures")
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    ok = all(q.correct_answer for q in result.questions)
    print(f"\nParsed {len(result.questions)} questions; "
          f"answer key matched for all: {ok}")


if __name__ == "__main__":
    main()
