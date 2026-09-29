"""Classical NLP pre-processing applied before any LLM call.

Cleaning and de-duplicating locally is free, and every removed character or
duplicate review is a token we do not pay for.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")

TEXT_COLUMN_HINTS = ("review", "text", "comment", "content", "body", "feedback")
RATING_COLUMN_HINTS = ("rating", "stars", "score")


@dataclass(frozen=True)
class Review:
    id: int
    text: str
    rating: float | None = None


def clean_text(text: str, max_chars: int) -> str:
    """Normalise unicode, strip HTML/URLs, collapse whitespace, truncate."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = _URL_RE.sub("[link]", _TAG_RE.sub(" ", text))
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0]
        text = f"{cut} ..."
    return text


def guess_column(columns: list[str], hints: tuple[str, ...]) -> str | None:
    """Pick the first column whose name contains a hint, ignoring ID columns."""
    candidates = [c for c in columns if not _is_id_column(c.lower())]
    for hint in hints:
        for column in candidates:
            if hint in column.lower():
                return column
    return None


def _is_id_column(name: str) -> bool:
    return name == "id" or name.endswith(("_id", " id", "-id"))


def prepare_reviews(
    df: pd.DataFrame,
    text_column: str,
    rating_column: str | None = None,
    *,
    max_chars: int = 1500,
    min_chars: int = 3,
    max_reviews: int = 500,
) -> tuple[list[Review], dict]:
    """Turn a raw dataframe into clean, unique Review objects.

    Returns the reviews and a small report of what was filtered out.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in data")

    texts = df[text_column].fillna("").astype(str).map(lambda t: clean_text(t, max_chars))
    ratings = (
        pd.to_numeric(df[rating_column], errors="coerce")
        if rating_column and rating_column in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    stats = {"input_rows": len(df), "empty_or_short": 0, "duplicates": 0, "truncated_to_cap": 0}
    seen: set[str] = set()
    reviews: list[Review] = []
    for text, rating in zip(texts, ratings):
        if len(text) < min_chars:
            stats["empty_or_short"] += 1
            continue
        fingerprint = _WS_RE.sub(" ", text.lower())
        if fingerprint in seen:
            stats["duplicates"] += 1
            continue
        seen.add(fingerprint)
        reviews.append(Review(id=len(reviews) + 1, text=text, rating=None if pd.isna(rating) else float(rating)))

    if len(reviews) > max_reviews:
        stats["truncated_to_cap"] = len(reviews) - max_reviews
        reviews = reviews[:max_reviews]
    stats["kept"] = len(reviews)
    return reviews, stats
