# Contributing

This project uses a branch and pull-request workflow.

## Workflow

1. **Pick or open an issue** describing the task (dataset, model, testing, documentation or deployment).
2. **Create a branch** from `main`, named after the type of work:
   ```bash
   git checkout main && git pull
   git checkout -b feature/<short-description>   # or fix/, docs/, test/
   ```
3. **Commit small, meaningful changes** with clear messages, for example `Add BM25 retrieval for Q&A`.
4. **Run the tests** before pushing:
   ```bash
   pytest -q
   ```
5. **Push and open a pull request** into `main`. Fill in the PR template and link the issue with `Closes #<number>`.
6. **Review**: at least one reviewer approves before merging.
7. **Merge** the approved PR and delete the branch.

## Code guidelines

- Keep LLM prompts in `prompts/prompts.yaml` and settings in `config/config.yaml`, never hard-coded.
- Never commit API keys. Use `.env` locally and Streamlit secrets when deployed.
- Add or update tests in `tests/` for any change to the pipeline.
