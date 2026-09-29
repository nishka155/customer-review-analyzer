"""Offline tests - a fake LLM client means no API key or network is needed."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from review_analyzer.aggregation import aspect_table, overall_stats, rating_mismatches
from review_analyzer.analyzer import ReviewAnalyzer
from review_analyzer.config import PROJECT_ROOT, load_config
from review_analyzer.llm_client import LLMError, ResponseCache
from review_analyzer.preprocessing import clean_text, guess_column, prepare_reviews
from review_analyzer.prompts import PromptLibrary
from review_analyzer.retrieval import BM25Index
from review_analyzer.schemas import BatchAnalysis, InsightReport, ReviewAnalysis

CONFIG = load_config()
PROMPTS = PromptLibrary.from_file(PROJECT_ROOT / CONFIG.paths.prompts)


def fake_analysis(review_id: int, text: str) -> ReviewAnalysis:
    negative = any(w in text.lower() for w in ("bad", "broke", "terrible"))
    return ReviewAnalysis(
        id=review_id,
        sentiment="negative" if negative else "positive",
        score=-0.8 if negative else 0.8,
        aspects=[{"aspect": "Product  Quality", "sentiment": "negative" if negative else "positive", "evidence": text[:20]}],
        emotions=["Frustration"] if negative else ["joy"],
        key_phrases=["earbuds"],
        summary=text[:40],
        actionable=negative,
        language="en",
    )


class FakeClient:
    """Parses review tags out of the prompt and answers deterministically."""

    def __init__(self, skip_ids: set[int] = frozenset(), fail: bool = False):
        self.calls: list[str] = []
        self.skip_ids = set(skip_ids)
        self.fail = fail

    def generate_json(self, system, prompt, schema):
        self.calls.append(prompt)
        if self.fail:
            raise LLMError("boom")
        if schema is BatchAnalysis:
            items = re.findall(r'<review id="(\d+)">(.*?)</review>', prompt)
            results = [fake_analysis(int(i), t) for i, t in items if int(i) not in self.skip_ids]
            self.skip_ids.clear()  # only skip on the first attempt
            return BatchAnalysis(results=results + [fake_analysis(999, "hallucinated")])
        if schema is InsightReport:
            return InsightReport(executive_summary="ok", strengths=[], pain_points=[],
                                 recommendations=[], emerging_issues=[])
        raise AssertionError(schema)

    def generate_text(self, system, prompt):
        self.calls.append(prompt)
        return "answer [#1]"


def make_reviews(texts, ratings=None):
    df = pd.DataFrame({"text": texts, "rating": ratings or [None] * len(texts)})
    return prepare_reviews(df, "text", "rating")[0]


# ------------------------------------------------------------ preprocessing

def test_clean_text_strips_html_urls_and_whitespace():
    assert clean_text("<b>Great</b>   buds &amp; case  see https://x.io", 100) == "Great buds & case see [link]"


def test_clean_text_truncates_on_word_boundary():
    assert clean_text("one two three four", 10) == "one two ..."


def test_prepare_reviews_dedupes_and_drops_empty():
    df = pd.DataFrame({"review": ["Good", "good ", "", None, "Bad sound"], "stars": [5, 5, 3, 1, "x"]})
    reviews, stats = prepare_reviews(df, "review", "stars")
    assert [r.text for r in reviews] == ["Good", "Bad sound"]
    assert [r.id for r in reviews] == [1, 2]
    assert reviews[1].rating is None
    assert stats["duplicates"] == 1 and stats["empty_or_short"] == 2


def test_guess_column():
    assert guess_column(["id", "Review_Text", "Rating"], ("review", "text")) == "Review_Text"
    assert guess_column(["review_id", "rating", "review_text"], ("review",)) == "review_text"
    assert guess_column(["a", "b"], ("review",)) is None


# ------------------------------------------------------------------ prompts

def test_prompts_render_without_missing_variables():
    system, user = PROMPTS["review_analysis"].render(
        {"aspects": "x", "new_aspect_rule": "y"}, product_context="p", count=1, reviews="r"
    )
    assert "$" not in system and "$" not in user
    for task, kwargs in {
        "insight_synthesis": dict(product_context="", total=1, stats="{}", aspects="[]",
                                  negative_quotes="", positive_quotes=""),
        "review_qa": dict(product_context="", stats="{}", reviews="", question="q"),
    }.items():
        PROMPTS[task].render(**kwargs)


def test_prompt_missing_variable_fails_loudly():
    with pytest.raises(KeyError):
        PROMPTS["review_qa"].render(question="q")


# ----------------------------------------------------------------- analyzer

def test_batch_analysis_batches_and_filters_hallucinated_ids():
    client = FakeClient()
    reviews = make_reviews([f"review number {i}" for i in range(23)])
    run = ReviewAnalyzer(client, PROMPTS, CONFIG).analyze(reviews)
    assert len(client.calls) == -(-23 // CONFIG.analysis.batch_size)  # ceil division
    assert sorted(run.analyses) == list(range(1, 24))
    assert run.failed_ids == []


def test_skipped_reviews_are_recovered():
    client = FakeClient(skip_ids={2})
    run = ReviewAnalyzer(client, PROMPTS, CONFIG).analyze(make_reviews(["a good one", "a bad one"]))
    assert set(run.analyses) == {1, 2}
    assert len(client.calls) == 2


def test_failed_batches_are_reported_not_raised():
    run = ReviewAnalyzer(FakeClient(fail=True), PROMPTS, CONFIG).analyze(make_reviews(["abc", "def"]))
    assert run.analyses == {} and run.failed_ids == [1, 2]


def test_synthesis_and_qa_use_retrieved_context():
    client = FakeClient()
    analyzer = ReviewAnalyzer(client, PROMPTS, CONFIG)
    run = analyzer.analyze(make_reviews(["battery life is great", "case hinge broke", "sound is bad"], [5, 1, 2]))
    assert analyzer.synthesize(run).executive_summary == "ok"
    assert analyzer.answer("What about the hinge?", run) == "answer [#1]"
    assert "[#2]" in client.calls[-1] and "battery life is great" not in client.calls[-1]


# -------------------------------------------------------------- aggregation

def test_aggregation():
    reviews = make_reviews(["great", "terrible", "bad but fine"], [5, 5, 1])
    analyses = {r.id: fake_analysis(r.id, r.text) for r in reviews}
    table = aspect_table(analyses)
    assert table.loc[0, "aspect"] == "product quality"  # normalised
    assert table.loc[0, "mentions"] == 3
    assert table.loc[0, "net_sentiment"] == pytest.approx(-1 / 3, abs=0.01)

    from review_analyzer.aggregation import build_results_frame
    df = build_results_frame(reviews, analyses)
    stats = overall_stats(df)
    assert stats["sentiment_counts"]["negative"] == 2
    assert stats["top_emotions"] == {"frustration": 2, "joy": 1}
    assert list(rating_mismatches(df)["id"]) == [2]  # 5 stars but negative text


# -------------------------------------------------------- retrieval & cache

def test_bm25_ranks_relevant_documents_first():
    index = BM25Index({1: "battery drains fast", 2: "great sound", 3: "battery battery life"})
    assert index.search("battery life", 2) == [3, 1]
    assert index.search("unrelated words", 5) == []


def test_response_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path)
    key = ResponseCache.key("model", "prompt")
    assert cache.get(key) is None
    cache.set(key, '{"a": 1}')
    assert cache.get(key) == '{"a": 1}'
