"""Command-line interface.

Example:
    python -m review_analyzer.cli data/sample_reviews.csv --product "AuraBuds Pro wireless earbuds"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .analyzer import build_analyzer
from .config import DEFAULT_CONFIG_PATH, get_api_key, load_config
from .llm_client import LLMError
from .preprocessing import RATING_COLUMN_HINTS, TEXT_COLUMN_HINTS, guess_column, prepare_reviews
from .report import to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-powered customer review analyzer")
    parser.add_argument("csv", type=Path, help="CSV file containing reviews")
    parser.add_argument("--text-column", help="column with review text (auto-detected if omitted)")
    parser.add_argument("--rating-column", help="optional star-rating column (auto-detected)")
    parser.add_argument("--product", default="", help="short product description for context")
    parser.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    api_key = get_api_key()
    if not api_key:
        print("Error: set GEMINI_API_KEY in your environment or .env file.", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    text_col = args.text_column or guess_column(list(df.columns), TEXT_COLUMN_HINTS)
    rating_col = args.rating_column or guess_column(list(df.columns), RATING_COLUMN_HINTS)
    if not text_col:
        print(f"Error: could not detect a text column in {list(df.columns)}; use --text-column.", file=sys.stderr)
        return 2

    a = config.analysis
    reviews, prep = prepare_reviews(
        df, text_col, rating_col,
        max_chars=a.max_review_chars, min_chars=a.min_review_chars, max_reviews=a.max_reviews,
    )
    print(f"Loaded {prep['input_rows']} rows -> {prep['kept']} clean, unique reviews")

    analyzer = build_analyzer(config, api_key)
    try:
        run = analyzer.analyze(
            reviews, args.product,
            progress=lambda done, total: print(f"\r  analysed {done}/{total}", end="", flush=True),
        )
        print()
        report = analyzer.synthesize(run, args.product)
    except LLMError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    run.frame.to_csv(args.out / "review_analysis.csv", index=False)
    run.aspects.to_csv(args.out / "aspect_summary.csv", index=False)
    (args.out / "insight_report.md").write_text(
        to_markdown(report, run.stats, run.aspects, args.product), encoding="utf-8"
    )

    usage = analyzer.client.usage
    print(f"Done in {run.elapsed_seconds:.1f}s | {usage.requests} API calls, {usage.cache_hits} cache hits, "
          f"{usage.total_tokens:,} tokens | failed reviews: {len(run.failed_ids)}")
    print(f"Results written to {args.out.resolve()}")
    print("\n" + report.executive_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
