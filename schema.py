"""
schema.py — the structured JSON schema for a parsed exam.

These dataclasses ARE the contract. On a real engagement we map them to the
client's predefined schema 1:1 (field renames are a config change, not a rewrite).
asdict() on ExamDocument yields production-ready JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Option:
    label: str            # "A", "B", ...
    text: str


@dataclass
class FigureRef:
    path: str             # relative path to the cropped PNG
    page: int
    bbox: list            # [x0, y0, x1, y1] in PDF points


@dataclass
class Question:
    number: int
    stem: str
    page: int
    options: list = field(default_factory=list)        # list[Option]
    figures: list = field(default_factory=list)        # list[FigureRef]
    correct_answer: Optional[str] = None               # "C"
    question_type: str = "mcq"                          # mcq | open | comprehension
    confidence: float = 0.0                             # 0-1, drives VLM fallback
    source_method: str = "deterministic"               # deterministic | vlm


@dataclass
class ExamDocument:
    source: str
    page_count: int
    questions: list = field(default_factory=list)       # list[Question]
    schema_version: str = "1.0"
