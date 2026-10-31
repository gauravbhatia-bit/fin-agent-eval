# fin-agent-eval

A local-first evaluation harness for a finance-education chatbot using fictional demo data. This is a prototype for evaluating AI agent reliability, not a financial product. It does not provide investment advice and is not affiliated with any broker.

## Status

Phases 1-3 are implemented. Phase 3 adds the record/replay/live Gemini client, hybrid BM25/embedding retrieval, guarded LangGraph agent, and `POST /chat`. Phase 4 is underway: it adds reviewable golden cases, metrics, judge auditing, resumable evaluation runs, a quality gate, and report comparison. Later phases will add the UI and deployment workflows. No model names or API keys are configured, so live Gemini requests have not been tested.

## Requirements

- Python 3.12
- Git
- Docker Desktop for later container work

On Windows, use the Python launcher to select Python 3.12:

```powershell
py -3.12 tasks.py setup
py -3.12 tasks.py lint
py -3.12 tasks.py test
py -3.12 tasks.py eval-smoke
```

The setup command creates a local `.venv`, installs the pinned dependencies from `requirements.txt`, and installs the pre-commit hooks. It does not require an API key.

After changing dependencies, regenerate the cross-platform lock with:

```powershell
py -3.12 -m piptools compile --allow-unsafe --generate-hashes --extra dev --output-file requirements.txt pyproject.toml
```

Run the API locally after setup:

```powershell
.\.venv\Scripts\python.exe tasks.py run-api
```

Then open `http://127.0.0.1:8000/health` or `http://127.0.0.1:8000/version`.

## Fictional pack and tools

The `packs/eu_us_etf/` files contain invented DemoBroker fees, instruments, and dated currency rates. They are for software tests only and are not current real-world data. Its 18 knowledge documents are unverified drafts; educational descriptions marked `TODO(verify)` need owner review before being treated as verified.

The tool registry in `app/tools/registry.py` exposes fee, hypothetical-growth, currency-conversion, instrument-lookup, position-size, knowledge-search, and simulated-escalation tools. Each request is validated with Pydantic. Invalid arguments return an explicit error. `POST /chat` applies policy rules before LLM calls, uses a manually bounded tool loop, and validates document citations. In-memory sessions retain up to six turns for 30 minutes and are lost on restart. The hybrid retriever caches document embeddings locally, separated by embedding model.

## Gemini models and quotas

The Gemini API model identifiers documented when Phase 3 started include `gemini-3.8-flash` and `gemini-embedding-2`. Model names are configuration values, not code constants. Free-tier access is limited to certain models, and per-project RPM/TPM/RPD limits vary; check your active limits in [AI Studio](https://aistudio.google.com/rate-limit?timeRange=last-28-days) and the official [models](https://ai.google.dev/gemini-api/docs/models), [pricing](https://ai.google.dev/gemini-api/docs/pricing), and [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) pages before setting `.env`. Free-tier requests may be used to improve Google's products; use only synthetic content.

## Configuration

Copy `.env.example` to `.env` to configure the application. Do not commit `.env` or paste API keys into source files. Model names are intentionally blank until checked against current Google AI Studio documentation.

## Safety and limitations

All broker data planned for this project is fictional and will be labelled as such. The application is educational only; it will not place trades, connect to brokerages, make predictions, or provide personalised investment, tax, or legal advice. Knowledge documents are a demo corpus and must not be treated as verified financial guidance.

The prototype has no authentication and uses per-process rate limits, a daily call counter, and in-memory sessions. A live model run requires a separately configured Gemini API key and model IDs, and may incur quota usage. Replay mode is intended for offline deterministic tests; it does not measure live model quality. Tests use fake model backends and do not require credentials or network access. The golden cases are drafts and require owner review; the evaluation runner refuses to run them unless `--allow-draft` is explicitly supplied. Live/replay evaluation runs need a Gemini key or a complete set of matching cassettes.

## Evaluation harness (Phase 4)

The evaluation interfaces are `evals.cases`, `evals.metrics`, `evals.judge`,
`evals.audit`, `evals.runner`, `evals.gate`, and `evals.compare`. Cases are loaded
from `evals/golden/eu_us_etf.jsonl`; they are not approved reference answers.
Gaurav must review and approve every case before a baseline-quality run. For
development only, include `--allow-draft`.

```powershell
.\.venv\Scripts\python.exe -m evals.runner --subset smoke --runs 1 --mode replay --allow-draft
.\.venv\Scripts\python.exe -m evals.runner --subset full --runs 3 --mode record --allow-draft
.\.venv\Scripts\python.exe -m evals.gate reports/latest.json
.\.venv\Scripts\python.exe -m evals.compare reports/before.json reports/latest.json
```

`py -3.12 tasks.py eval-smoke`, `eval-record`, `eval-full`, and `gate` are
convenience wrappers. `--mode live` and `record` use the configured Gemini API key and can consume
quota. Replay mode never uses the network and raises a clear error if any needed
generation or embedding cassette is missing. `--resume` keeps completed
case/run pairs in the output report. Baseline targets are placeholders, not
measured scores. Manually reviewed judge labels belong in
`evals/human_labels.jsonl`; the audit reports unavailable until labels are
provided. `GET /eval/latest` returns a summary of `reports/latest.json` when it
exists.

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for phase scope, acceptance criteria, deployment requirements, and known limitations.

## GitHub setup

The CI workflow runs lint and tests on pushes to `main` and pull requests. Configure branch protection after the CI checks exist; deployment credentials and external accounts are only needed in later phases.
