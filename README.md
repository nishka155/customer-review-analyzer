# 📊 Customer Review Analyzer

An NLP application that turns hundreds of raw customer reviews into a decision-ready product report, using **Google Gemini** through its API.

It does three LLM-powered NLP tasks:

| Task | What it does | LLM technique |
|---|---|---|
| **Aspect-based sentiment analysis** | For every review: overall sentiment and score, per-aspect opinions with quoted evidence, emotions, key phrases, one-line summary, "actionable" flag, language detection | Batched structured output (JSON schema) with a few-shot example |
| **Insight synthesis** | Executive summary, ranked strengths and pain points, prioritised recommendations, emerging (safety/defect) issues | Grounded summarisation over statistics computed in Python and representative quotes |
| **Ask the reviews (Q&A)** | Answers free-text questions such as *"What do people say about the battery?"*, citing review IDs | Retrieval-augmented generation (BM25 retrieval + citations) |

The app also includes a **single-review analyzer**, **rating vs. text contradiction detection**, interactive charts, and CSV/Markdown export.

---

## Architecture

```
CSV ──► preprocessing ──► batch analysis (Gemini, parallel) ──► aggregation (pandas)
        clean, dedupe,     10 reviews per call, JSON schema,    exact stats, aspect table,
        truncate           retry + recovery, disk cache         rating mismatches
                                                                   │
                        ┌──────────────────────────────────────────┤
                        ▼                                          ▼
              insight synthesis (Gemini)                 Q&A: BM25 top-k ► Gemini
              stats + quotes ► report                    answer with [#id] citations
```

```
customer-review-analyzer/
├── app.py                      # Streamlit web UI
├── config/config.yaml          # Configuration file (model, batching, cache, aspects...)
├── prompts/prompts.yaml        # Prompt file (all system prompts + templates)
├── data/sample_reviews.csv     # 40 sample reviews (includes edge cases)
├── review_analyzer/
│   ├── config.py               # Typed config loading (pydantic) + API key from .env
│   ├── prompts.py              # Prompt library loader / renderer
│   ├── schemas.py              # Output schemas (sent to Gemini and used for validation)
│   ├── llm_client.py           # Gemini client: retries, JSON mode, caching, token usage
│   ├── preprocessing.py        # Text cleaning, deduplication, column detection
│   ├── analyzer.py             # Orchestration of the 3 LLM tasks
│   ├── aggregation.py          # Deterministic statistics
│   ├── retrieval.py            # BM25 index for Q&A grounding
│   ├── report.py               # Markdown report export
│   └── cli.py                  # Command-line interface
├── tests/test_pipeline.py      # 13 offline tests (fake LLM client, no key needed)
├── requirements.txt
└── .env.example
```

---

## Setup

**Requirements:** Python 3.10 or later and a free Gemini API key from <https://aistudio.google.com/apikey>.

```bash
git clone <your-repo-url>
cd customer-review-analyzer
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

### Run the web app

```bash
streamlit run app.py
```

Open <http://localhost:8501>, keep **Sample dataset** selected, and click **Analyze reviews**.

### Run from the command line

```bash
python -m review_analyzer.cli data/sample_reviews.csv --product "AuraBuds Pro wireless earbuds"
```

This writes `output/review_analysis.csv`, `output/aspect_summary.csv` and `output/insight_report.md`.

### Run tests

```bash
pytest -q
```

The tests use a fake LLM client, so they need no API key or network.

---

## Prompt design (`prompts/prompts.yaml`)

* **Separate system and user prompts for each task.** The system prompt holds the role, rules and output contract. The user template holds only the data. This makes the prompts reusable and easy to version (`version:` field).
* **Two layers of structure.** The prompt explains what each field *means*: sentiment definitions, score ranges, and evidence limited to 12 verbatim words. A Pydantic JSON schema passed to Gemini (`response_schema`) enforces the *shape*, so the output is always parseable.
* **One compact few-shot example.** It shows the hardest cases: a *mixed* review, the same aspect with both polarities, and mapping onto canonical aspects.
* **Canonical aspect list** (set in the config). Opinions are mapped onto consistent labels, so aspect charts don't split into synonyms like "battery" and "battery life". A config switch controls whether the model may add new aspects.
* **Prompt-injection defence.** Reviews are wrapped in `<review id=..>` tags and declared untrusted data. The sample data includes a review saying *"Ignore all previous instructions..."* to demonstrate this.
* **Grounding rules.** The synthesis step is told never to invent numbers, because every statistic is computed in Python and passed in. Q&A must cite `[#id]` and say so when the reviews don't contain the answer.
* **Sarcasm and multilingual handling** are covered by explicit rules. Summaries are always in English and the original language is recorded.
* Templates use `$placeholders` (`string.Template`), so JSON braces never need escaping. A missing variable raises an error instead of silently sending a broken prompt.

## Efficiency

* **Batching:** 10 reviews per request, which means about 10× fewer API calls than one call per review. The size is configurable.
* **Concurrency:** batches run in parallel through a thread pool (`max_workers`).
* **Token savings:** HTML and URLs are stripped, whitespace is collapsed, exact duplicates are removed, long reviews are truncated, and "thinking" is disabled for the labelling task. Synthesis receives aggregated stats and at most 25+25 quotes, not the whole dataset. Q&A sends only the top-k retrieved reviews.
* **Disk cache:** identical requests are served from `.cache/`, so a re-run costs nothing.
* **Resilience:** exponential backoff with jitter on 429/5xx errors and on malformed JSON. Reviews the model skips are re-requested once. Failed batches are reported without crashing the run, and hallucinated review IDs are discarded.
* **Usage tracking:** API calls, cache hits and tokens are shown in the UI and CLI.

## Configuration (`config/config.yaml`)

Key settings: `llm.model`, `temperature`, `thinking_budget`, `max_retries`, `analysis.batch_size`, `max_workers`, `aspects`, `allow_new_aspects`, `cache.enabled`, and `qa.top_k_reviews`. The API key is **never** stored in the config. It is read from the `GEMINI_API_KEY` environment variable or `.env`, which is git-ignored.

## Input format

Any CSV with one review per row. The text column and an optional star-rating column are auto-detected (for example `review`, `text`, `comment`, `rating`, `stars`), and you can pick them manually in the UI or with `--text-column` and `--rating-column`.
