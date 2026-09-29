"""Deterministic aggregation of per-review LLM output into statistics.

Numbers are computed in Python, not by the LLM, so they are exact; the LLM is
only asked to interpret them in the synthesis step.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .preprocessing import Review
from .schemas import ReviewAnalysis

SENTIMENTS = ("positive", "neutral", "mixed", "negative")


def build_results_frame(reviews: list[Review], analyses: dict[int, ReviewAnalysis]) -> pd.DataFrame:
    """One row per successfully analysed review."""
    rows = []
    for review in reviews:
        a = analyses.get(review.id)
        if a is None:
            continue
        rows.append(
            {
                "id": review.id,
                "review": review.text,
                "rating": review.rating,
                "sentiment": a.sentiment,
                "score": round(a.score, 2),
                "summary": a.summary,
                "aspects": "; ".join(f"{x.aspect} ({x.sentiment})" for x in a.aspects),
                "emotions": ", ".join(a.emotions),
                "key_phrases": ", ".join(a.key_phrases),
                "actionable": a.actionable,
                "language": a.language,
            }
        )
    return pd.DataFrame(rows)


def aspect_table(analyses: dict[int, ReviewAnalysis]) -> pd.DataFrame:
    """Mentions and net sentiment per aspect (counted once per review+polarity)."""
    counts: defaultdict[str, Counter] = defaultdict(Counter)
    reviews_per_aspect: Counter = Counter()
    for a in analyses.values():
        pairs = {(x.aspect, x.sentiment) for x in a.aspects}
        for aspect, polarity in pairs:
            counts[aspect][polarity] += 1
        reviews_per_aspect.update({aspect for aspect, _ in pairs})

    total = max(len(analyses), 1)
    rows = []
    for aspect, c in counts.items():
        mentions = c["positive"] + c["negative"] + c["neutral"]
        rows.append(
            {
                "aspect": aspect,
                "mentions": mentions,
                "positive": c["positive"],
                "negative": c["negative"],
                "neutral": c["neutral"],
                "net_sentiment": round((c["positive"] - c["negative"]) / mentions, 2),
                "pct_of_reviews": round(100 * reviews_per_aspect[aspect] / total, 1),
            }
        )
    columns = ["aspect", "mentions", "positive", "negative", "neutral", "net_sentiment", "pct_of_reviews"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("mentions", ascending=False, ignore_index=True)


def overall_stats(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Headline numbers for the dashboard, the insight prompt and Q&A."""
    if df.empty:
        return {"total": 0}
    total = len(df)
    counts = df["sentiment"].value_counts()
    stats = {
        "total": total,
        "sentiment_counts": {s: int(counts.get(s, 0)) for s in SENTIMENTS},
        "sentiment_pct": {s: round(100 * counts.get(s, 0) / total, 1) for s in SENTIMENTS},
        "avg_sentiment_score": round(float(df["score"].mean()), 3),
        "actionable_pct": round(100 * float(df["actionable"].mean()), 1),
        "top_emotions": _top_terms(df["emotions"], top_n),
        "top_key_phrases": _top_terms(df["key_phrases"], top_n),
        "languages": df["language"].value_counts().head(5).to_dict(),
    }
    rated = df.dropna(subset=["rating"])
    if not rated.empty:
        stats["avg_rating"] = round(float(rated["rating"].mean()), 2)
        stats["rating_text_mismatches"] = int(len(rating_mismatches(rated)))
    return stats


def rating_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews whose star rating contradicts the text (e.g. 5 stars, angry text)."""
    rated = df.dropna(subset=["rating"])
    high = (rated["rating"] >= 4) & (rated["sentiment"] == "negative")
    low = (rated["rating"] <= 2) & (rated["sentiment"] == "positive")
    return rated[high | low]


def _top_terms(series: pd.Series, top_n: int) -> dict[str, int]:
    counter: Counter = Counter()
    for cell in series.dropna():
        counter.update(t.strip() for t in str(cell).split(",") if t.strip())
    return dict(counter.most_common(top_n))
