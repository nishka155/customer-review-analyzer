## Summary

Describe what this pull request changes and why.
Example: *"Adds automatic model fallback so analysis keeps working when the primary Gemini model returns 503."*

**Linked issue:** Closes #<issue-number>

## Type of change

- [ ] Feature (new functionality, e.g. a new analysis tab)
- [ ] Bug fix (e.g. wrong column detected in uploaded CSV)
- [ ] Documentation (README, installation guide, presentation)
- [ ] Tests (new or updated cases in `tests/`)
- [ ] Deployment / configuration (`config/config.yaml`, Streamlit settings)

## How it was tested

- `pytest -q`: all tests pass
- Manual check: ran `streamlit run app.py` on `data/sample_reviews.csv` and confirmed the dashboard, Q&A and single-review tabs work

## Checklist

- [ ] `pytest -q` passes
- [ ] Prompts changed only in `prompts/prompts.yaml`, settings only in `config/config.yaml`
- [ ] No API keys or secrets committed (`.env` stays local)
- [ ] README or `docs/` updated if behaviour changed
