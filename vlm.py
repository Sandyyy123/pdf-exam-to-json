"""
vlm.py — Vision-Language-Model fallback adapter (GPT-4o / Claude).

Only called for blocks the deterministic PyMuPDF pass flags as low-confidence
(formula-heavy stems, fully graphical questions, scanned/handwritten pages).
This keeps the per-document API cost near zero across a large library: the
expensive model touches only the few percent of content that needs it.

Both providers take an image (a cropped PNG of the question region) plus a
prompt and return JSON text. Swap the provider with one env var.
"""
from __future__ import annotations

import os
import base64


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def gpt4o_extract(image_path: str, prompt: str) -> str:
    """OpenAI GPT-4o vision extraction. Returns raw JSON string."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
            ],
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return resp.choices[0].message.content


def claude_extract(image_path: str, prompt: str) -> str:
    """Anthropic Claude vision extraction. Returns raw JSON string."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": _b64(image_path)}},
                {"type": "text", "text": prompt + " Respond with JSON only."},
            ],
        }],
    )
    return resp.content[0].text


def get_vlm():
    """Return the configured VLM callable, or None to stay deterministic-only."""
    provider = os.environ.get("VLM_PROVIDER", "").lower()
    if provider == "openai":
        return gpt4o_extract
    if provider == "anthropic":
        return claude_extract
    return None
