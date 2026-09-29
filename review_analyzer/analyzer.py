"""Orchestrates the three LLM tasks: per-review analysis, insight synthesis and Q&A."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from .aggregation import aspect_table, build_results_frame, overall_stats
from .config import AppConfig
from .llm_client import GeminiClient, LLMClient, LLMError, ResponseCache
from .preprocessing import Review
from .prompts import PromptLibrary
from .retrieval import BM25Index
from .schemas import BatchAnalysis, InsightReport, ReviewAnalysis

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
QUOTE_CHARS = 300


@dataclass
class AnalysisRun:
    """Everything produced by one batch analysis."""

    reviews: list[Review]
    analyses: dict[int, ReviewAnalysis]
    failed_ids: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @cached_property
    def frame(self) -> pd.DataFrame:
        return build_results_frame(self.reviews, self.analyses)

    @cached_property
    def aspects(self) -> pd.DataFrame:
        return aspect_table(self.analyses)

    @cached_property
    def stats(self) -> dict:
        return overall_stats(self.frame)


class ReviewAnalyzer:
    def __init__(self, client: LLMClient, prompts: PromptLibrary, config: AppConfig):
        self.client = client
        self.prompts = prompts
        self.config = config

    # ------------------------------------------------------ 1. review analysis

    def analyze(
        self,
        reviews: list[Review],
        product_context: str = "",
        progress: ProgressCallback | None = None,
    ) -> AnalysisRun:
        """Analyse reviews in batches, concurrently, with per-review recovery."""
        start = time.perf_counter()
        size = self.config.analysis.batch_size
        batches = [reviews[i : i + size] for i in range(0, len(reviews), size)]
        analyses: dict[int, ReviewAnalysis] = {}
        failed: list[int] = []
        done = 0

        with ThreadPoolExecutor(max_workers=self.config.analysis.max_workers) as pool:
            futures = {pool.submit(self._analyze_with_recovery, b, product_context): b for b in batches}
            for future in as_completed(futures):
                batch_results, batch_failed = future.result()
                analyses.update(batch_results)
                failed.extend(batch_failed)
                done += len(futures[future])
                if progress:
                    progress(done, len(reviews))

        return AnalysisRun(reviews, analyses, sorted(failed), time.perf_counter() - start)

    def analyze_single(self, text: str, product_context: str = "") -> ReviewAnalysis:
        results = self._analyze_batch([Review(id=1, text=text)], product_context)
        if 1 not in results:
            raise LLMError("The model did not return an analysis for this review.")
        return results[1]

    def _analyze_with_recovery(
        self, batch: list[Review], product_context: str
    ) -> tuple[dict[int, ReviewAnalysis], list[int]]:
        """Run one batch; re-ask once for any reviews the model skipped."""
        try:
            results = self._analyze_batch(batch, product_context)
        except LLMError as exc:
            log.error("Batch %s failed: %s", [r.id for r in batch], exc)
            return {}, [r.id for r in batch]

        missing = [r for r in batch if r.id not in results]
        if missing:
            log.warning("Model skipped %d review(s); retrying them", len(missing))
            try:
                results.update(self._analyze_batch(missing, product_context))
            except LLMError as exc:
                log.error("Recovery batch failed: %s", exc)
        return results, [r.id for r in batch if r.id not in results]

    def _analyze_batch(self, batch: list[Review], product_context: str) -> dict[int, ReviewAnalysis]:
        cfg = self.config.analysis
        system, prompt = self.prompts["review_analysis"].render(
            {
                "aspects": ", ".join(cfg.aspects) or "(none - infer aspects)",
                "new_aspect_rule": self.prompts.new_aspect_rule(cfg.allow_new_aspects),
            },
            product_context=product_context or "Not specified",
            count=len(batch),
            reviews="\n".join(f'<review id="{r.id}">{r.text}</review>' for r in batch),
        )
        output = self.client.generate_json(system, prompt, BatchAnalysis)
        wanted = {r.id for r in batch}
        # Ignore any hallucinated ids; keep the first result per id.
        results: dict[int, ReviewAnalysis] = {}
        for item in output.results:
            if item.id in wanted and item.id not in results:
                results[item.id] = item
        return results

    # ----------------------------------------------------- 2. insight synthesis

    def synthesize(self, run: AnalysisRun, product_context: str = "") -> InsightReport:
        """Turn aggregated statistics + representative quotes into a report."""
        df = run.frame
        if df.empty:
            raise LLMError("No analysed reviews to synthesise.")
        n = self.config.synthesis.max_quotes_per_polarity
        negative = df[df["sentiment"].isin(["negative", "mixed"])].nsmallest(n, "score")
        positive = df[df["sentiment"].isin(["positive", "mixed"])].nlargest(n, "score")

        system, prompt = self.prompts["insight_synthesis"].render(
            product_context=product_context or "Not specified",
            total=len(df),
            stats=json.dumps(run.stats, ensure_ascii=False),
            aspects=run.aspects.to_json(orient="records"),
            negative_quotes=_format_quotes(negative),
            positive_quotes=_format_quotes(positive),
        )
        return self.client.generate_json(system, prompt, InsightReport)

    # --------------------------------------------------------------- 3. Q & A

    def answer(self, question: str, run: AnalysisRun, product_context: str = "") -> str:
        """Retrieval-augmented answer grounded in the analysed reviews."""
        df = run.frame
        if df.empty:
            raise LLMError("No analysed reviews to answer questions about.")
        documents = {
            int(row.id): f"{row.review} {row.summary} {row.aspects} {row.key_phrases}"
            for row in df.itertuples()
        }
        top_k = self.config.qa.top_k_reviews
        ids = BM25Index(documents).search(question, top_k)
        if not ids:  # no lexical overlap (e.g. "overall, how is it?") -> spread sample
            ids = df.sample(min(top_k, len(df)), random_state=0)["id"].tolist()
        context = df.set_index("id").loc[ids].reset_index()

        system, prompt = self.prompts["review_qa"].render(
            product_context=product_context or "Not specified",
            stats=json.dumps(run.stats, ensure_ascii=False),
            reviews=_format_quotes(context),
            question=question.strip(),
        )
        return self.client.generate_text(system, prompt)


def build_analyzer(config: AppConfig, api_key: str) -> ReviewAnalyzer:
    """Wire the Gemini client, cache and prompt library together."""
    cache = ResponseCache(config.resolve(config.cache.directory)) if config.cache.enabled else None
    client = GeminiClient(api_key, config.llm, cache)
    prompts = PromptLibrary.from_file(config.resolve(config.paths.prompts))
    return ReviewAnalyzer(client, prompts, config)


def _format_quotes(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    lines = []
    for row in df.itertuples():
        text = row.review if len(row.review) <= QUOTE_CHARS else row.review[:QUOTE_CHARS] + " ..."
        rating = f", rating {row.rating:g}" if pd.notna(row.rating) else ""
        lines.append(f'[#{row.id}] ({row.sentiment}, score {row.score:+.2f}{rating}) "{text}"')
    return "\n".join(lines)
