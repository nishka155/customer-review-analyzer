"""Render an insight report + statistics as a Markdown document."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .schemas import InsightReport


def to_markdown(report: InsightReport, stats: dict, aspects: pd.DataFrame, product_context: str = "") -> str:
    lines = [
        "# Customer Review Insight Report",
        "",
        f"*Product:* {product_context or 'Not specified'}  ",
        f"*Reviews analysed:* {stats.get('total', 0)}  ",
        f"*Generated:* {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Executive summary",
        report.executive_summary,
        "",
        "## Sentiment overview",
        "| Sentiment | Reviews | Share |",
        "|---|---:|---:|",
    ]
    for label, count in stats.get("sentiment_counts", {}).items():
        lines.append(f"| {label} | {count} | {stats['sentiment_pct'][label]}% |")
    lines += ["", f"- Average sentiment score: **{stats.get('avg_sentiment_score', 0):+.2f}** (-1 to +1)"]
    if "avg_rating" in stats:
        lines.append(f"- Average star rating: **{stats['avg_rating']}**")
        lines.append(f"- Reviews where rating contradicts text: **{stats['rating_text_mismatches']}**")

    lines += ["", "## Aspect breakdown", "| Aspect | Mentions | + | - | Net |", "|---|---:|---:|---:|---:|"]
    for row in aspects.itertuples():
        lines.append(f"| {row.aspect} | {row.mentions} | {row.positive} | {row.negative} | {row.net_sentiment:+.2f} |")

    for title, themes in (("Strengths", report.strengths), ("Pain points", report.pain_points)):
        lines += ["", f"## {title}"]
        for t in themes:
            refs = ", ".join(f"#{i}" for i in t.example_review_ids)
            lines.append(f"- **{t.title}** ({t.frequency} frequency) - {t.description} _(reviews {refs})_")

    lines += ["", "## Recommendations"]
    for i, r in enumerate(report.recommendations, 1):
        lines.append(f"{i}. **[{r.priority.upper()}] {r.action}** - {r.rationale} _(aspect: {r.related_aspect})_")

    lines += ["", "## Emerging issues"]
    lines += [f"- {issue}" for issue in report.emerging_issues] or ["- None detected."]
    return "\n".join(lines) + "\n"
