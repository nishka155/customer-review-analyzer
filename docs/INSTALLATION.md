# Installation Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or newer | <https://www.python.org/downloads/> |
| Git | any | <https://git-scm.com/downloads> |
| Gemini API key | - | Free at <https://aistudio.google.com/apikey> |

## 1. Clone the repository

```bash
git clone https://github.com/nishka155/customer-review-analyzer.git
cd customer-review-analyzer
```

## 2. Create a virtual environment and install dependencies

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Add your API key

Copy the template and paste your key into it:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

`.env` should contain:

```
GEMINI_API_KEY=your-api-key-here
```

`.env` is listed in `.gitignore`, so the key is never committed.

## 4. Run

| Mode | Command | Result |
|---|---|---|
| Web app | `streamlit run app.py` | Opens <http://localhost:8501> |
| Command line | `python -m review_analyzer.cli data/sample_reviews.csv --product "AuraBuds Pro wireless earbuds"` | Writes CSV and Markdown files to `output/` |
| Tests | `pytest -q` | 13 offline tests; no API key needed |

## 5. Deploy online (optional)

1. Sign in at <https://share.streamlit.io> with GitHub and click **Create app**.
2. Choose this repository, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, add `GEMINI_API_KEY = "your-key"`.
4. Click **Deploy**.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Enter a Gemini API key" shown | `.env` is missing or the variable name is wrong. On Streamlit Cloud, check **Settings → Secrets**. |
| `404 ... model is no longer available` | Change `llm.model` in `config/config.yaml`; the fallback models are tried automatically. |
| `429 RESOURCE_EXHAUSTED` | Free-tier quota reached. Wait a minute, or lower `analysis.max_workers` in the config. |
| `503 UNAVAILABLE` | The model is overloaded; the app switches to a fallback model automatically. |
| "No usable reviews found" | The wrong text column was selected. Pick the column containing review text. |
