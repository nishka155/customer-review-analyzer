"""Streamlit web UI for the Customer Review Analyzer.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from review_analyzer.aggregation import rating_mismatches
from review_analyzer.analyzer import AnalysisRun, ReviewAnalyzer, build_analyzer
from review_analyzer.config import get_api_key, load_config
from review_analyzer.llm_client import LLMError
from review_analyzer.preprocessing import RATING_COLUMN_HINTS, TEXT_COLUMN_HINTS, guess_column, prepare_reviews
from review_analyzer.report import to_markdown

SENTIMENT_COLORS = {"positive": "#2e9e5b", "neutral": "#9aa0a6", "mixed": "#e0a526", "negative": "#d64545"}

st.set_page_config(page_title="Customer Review Analyzer", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner=False)
def get_analyzer(api_key: str, model: str, batch_size: int) -> ReviewAnalyzer:
    config = load_config()
    config.llm.model = model
    config.analysis.batch_size = batch_size
    return build_analyzer(config, api_key)


def sidebar() -> tuple[ReviewAnalyzer | None, str]:
    config = load_config()
    with st.sidebar:
        st.header("Settings")
        env_key = get_api_key()
        api_key = env_key or st.text_input("Gemini API key", type="password", help="Or set GEMINI_API_KEY in .env")
        if env_key:
            st.success("API key loaded from environment", icon="🔑")
        model = st.text_input("Model", config.llm.model)
        batch_size = st.slider("Reviews per API call", 1, 25, config.analysis.batch_size)
        product = st.text_input("Product context", "AuraBuds Pro wireless earbuds",
                                help="Helps the model interpret domain-specific aspects")
        st.caption("Prompts: `prompts/prompts.yaml` · Config: `config/config.yaml`")

    if not api_key:
        st.info("Enter a Gemini API key in the sidebar (free at aistudio.google.com) to begin.")
        return None, product
    return get_analyzer(api_key, model, batch_size), product


def load_data(config) -> pd.DataFrame | None:
    source = st.radio("Data source", ["Sample dataset", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        file = st.file_uploader("CSV with one review per row", type="csv")
        return pd.read_csv(file) if file else None
    return pd.read_csv(config.resolve(config.paths.sample_data))


def batch_tab(analyzer: ReviewAnalyzer, product: str) -> None:
    config = analyzer.config
    df = load_data(config)
    if df is None:
        return
    columns = list(df.columns)
    c1, c2 = st.columns(2)
    guessed_text = guess_column(columns, TEXT_COLUMN_HINTS) or columns[0]
    text_col = c1.selectbox("Review text column", columns, index=columns.index(guessed_text))
    rating_options = ["(none)"] + columns
    guessed_rating = guess_column(columns, RATING_COLUMN_HINTS)
    rating_col = c2.selectbox("Star rating column (optional)", rating_options,
                              index=rating_options.index(guessed_rating) if guessed_rating else 0)
    with st.expander(f"Preview ({len(df)} rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    if st.button("Analyze reviews", type="primary"):
        a = config.analysis
        reviews, prep = prepare_reviews(
            df, text_col, None if rating_col == "(none)" else rating_col,
            max_chars=a.max_review_chars, min_chars=a.min_review_chars, max_reviews=a.max_reviews,
        )
        st.caption(f"Pre-processing: kept {prep['kept']} of {prep['input_rows']} rows "
                   f"(removed {prep['empty_or_short']} empty, {prep['duplicates']} duplicates)")
        bar = st.progress(0.0, text="Analysing reviews...")
        try:
            run = analyzer.analyze(reviews, product, progress=lambda d, t: bar.progress(d / t, text=f"Analysed {d}/{t}"))
            with st.spinner("Synthesising insights..."):
                report = analyzer.synthesize(run, product)
        except LLMError as exc:
            st.error(str(exc))
            return
        bar.empty()
        st.session_state.update(run=run, report=report, product=product)

    if "run" in st.session_state:
        render_results(st.session_state["run"], st.session_state["report"], st.session_state["product"], analyzer)


def render_results(run: AnalysisRun, report, product: str, analyzer: ReviewAnalyzer) -> None:
    df, stats, aspects = run.frame, run.stats, run.aspects
    usage = analyzer.client.usage

    m = st.columns(5)
    m[0].metric("Reviews analysed", stats["total"])
    m[1].metric("Positive", f"{stats['sentiment_pct']['positive']}%")
    m[2].metric("Negative", f"{stats['sentiment_pct']['negative']}%")
    m[3].metric("Avg. sentiment", f"{stats['avg_sentiment_score']:+.2f}")
    m[4].metric("Actionable", f"{stats['actionable_pct']}%")
    st.caption(f"{run.elapsed_seconds:.1f}s · {usage.requests} API calls · {usage.cache_hits} cache hits · "
               f"{usage.total_tokens:,} tokens used this session")
    if run.failed_ids:
        st.warning(f"{len(run.failed_ids)} review(s) could not be analysed: {run.failed_ids}")

    st.subheader("Executive summary")
    st.write(report.executive_summary)

    left, right = st.columns([1, 2])
    counts = pd.DataFrame(stats["sentiment_counts"].items(), columns=["sentiment", "reviews"])
    left.plotly_chart(
        px.pie(counts, names="sentiment", values="reviews", hole=0.5, color="sentiment",
               color_discrete_map=SENTIMENT_COLORS, title="Overall sentiment"),
        use_container_width=True,
    )
    if not aspects.empty:
        long = aspects.melt(id_vars="aspect", value_vars=["positive", "neutral", "negative"],
                            var_name="sentiment", value_name="mentions")
        right.plotly_chart(
            px.bar(long, y="aspect", x="mentions", color="sentiment", orientation="h",
                   color_discrete_map=SENTIMENT_COLORS, title="Sentiment by aspect",
                   category_orders={"aspect": aspects["aspect"].tolist()}),
            use_container_width=True,
        )

    s, p = st.columns(2)
    with s:
        st.subheader("Strengths")
        for t in report.strengths:
            st.markdown(f"**{t.title}** · _{t.frequency}_  \n{t.description}")
    with p:
        st.subheader("Pain points")
        for t in report.pain_points:
            st.markdown(f"**{t.title}** · _{t.frequency}_  \n{t.description}")

    st.subheader("Recommendations")
    st.dataframe(pd.DataFrame([r.model_dump() for r in report.recommendations]), use_container_width=True, hide_index=True)
    if report.emerging_issues:
        st.error("**Emerging issues:** " + " · ".join(report.emerging_issues))

    st.subheader("Review-level results")
    choice = st.multiselect("Filter by sentiment", list(SENTIMENT_COLORS), default=list(SENTIMENT_COLORS))
    st.dataframe(df[df["sentiment"].isin(choice)], use_container_width=True, hide_index=True)

    mismatches = rating_mismatches(df)
    if not mismatches.empty:
        with st.expander(f"⚠️ {len(mismatches)} review(s) where the star rating contradicts the text"):
            st.dataframe(mismatches[["id", "rating", "sentiment", "review"]], hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", df.to_csv(index=False), "review_analysis.csv", "text/csv")
    d2.download_button("Download report (Markdown)", to_markdown(report, stats, aspects, product),
                       "insight_report.md", "text/markdown")


def qa_tab(analyzer: ReviewAnalyzer) -> None:
    run: AnalysisRun | None = st.session_state.get("run")
    if run is None:
        st.info("Run a batch analysis first, then ask questions about the reviews here.")
        return
    history = st.session_state.setdefault("chat", [])
    for role, text in history:
        st.chat_message(role).markdown(text)
    if question := st.chat_input("e.g. What do customers say about battery life?"):
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Searching reviews..."):
            try:
                answer = analyzer.answer(question, run, st.session_state.get("product", ""))
            except LLMError as exc:
                answer = f"Error: {exc}"
            st.markdown(answer)
        history += [("user", question), ("assistant", answer)]


def single_tab(analyzer: ReviewAnalyzer, product: str) -> None:
    text = st.text_area("Paste a single review", height=150,
                        placeholder="The sound is great but the case hinge broke after a month...")
    if st.button("Analyze review") and text.strip():
        try:
            with st.spinner("Analysing..."):
                result = analyzer.analyze_single(text.strip(), product)
        except LLMError as exc:
            st.error(str(exc))
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", result.sentiment.title())
        c2.metric("Score", f"{result.score:+.2f}")
        c3.metric("Actionable", "Yes" if result.actionable else "No")
        st.write(f"**Summary:** {result.summary}")
        st.write(f"**Emotions:** {', '.join(result.emotions) or '-'}  ·  **Key phrases:** {', '.join(result.key_phrases)}")
        if result.aspects:
            st.dataframe(pd.DataFrame([a.model_dump() for a in result.aspects]), hide_index=True, use_container_width=True)


def main() -> None:
    st.title("📊 Customer Review Analyzer")
    st.caption("Aspect-based sentiment analysis, insight synthesis and review Q&A powered by Google Gemini")
    analyzer, product = sidebar()
    if analyzer is None:
        return
    tab_batch, tab_qa, tab_single = st.tabs(["Batch analysis", "Ask the reviews", "Single review"])
    with tab_batch:
        batch_tab(analyzer, product)
    with tab_qa:
        qa_tab(analyzer)
    with tab_single:
        single_tab(analyzer, product)


main()
