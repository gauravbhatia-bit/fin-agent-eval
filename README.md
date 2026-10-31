# fin-agent-eval

A local-first evaluation harness for a finance-education chatbot using fictional demo data. This is a prototype for evaluating AI agent reliability, not a financial product. It does not provide investment advice and is not affiliated with any broker.

## Status

Phases 1 and 2 are complete: the local scaffold, fictional `eu_us_etf` domain pack, and deterministic tool registry have passing tests. Later phases will add the chatbot, evaluation harness, UI, and deployment workflows. No model names have been configured yet.

## Requirements

- Python 3.12
- Git
- Docker Desktop for later container work

On Windows, use the Python launcher to select Python 3.12:

```powershell
py -3.12 tasks.py setup
py -3.12 tasks.py lint
py -3.12 tasks.py test
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

The tool registry in `app/tools/registry.py` exposes fee, hypothetical-growth, currency-conversion, instrument-lookup, position-size, knowledge-search, and simulated-escalation tools. Each request is validated with Pydantic. Invalid arguments return an explicit error. The Phase 2 knowledge search uses deterministic keyword overlap; Phase 3 will provide the hybrid retriever.

## Configuration

Copy `.env.example` to `.env` to configure the application. Do not commit `.env` or paste API keys into source files. Model names are intentionally blank until checked against current Google AI Studio documentation.

## Safety and limitations

All broker data planned for this project is fictional and will be labelled as such. The application is educational only; it will not place trades, connect to brokerages, make predictions, or provide personalised investment, tax, or legal advice. Knowledge documents are a demo corpus and must not be treated as verified financial guidance.

The prototype will have no authentication and will use in-memory sessions. A live model run requires a separately configured Gemini API key and may incur quota usage. Replay mode is intended for offline deterministic tests; it does not measure live model quality.

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for phase scope, acceptance criteria, deployment requirements, and known limitations.

## GitHub setup

The CI workflow runs lint and tests on pushes to `main` and pull requests. Configure branch protection after the CI checks exist; deployment credentials and external accounts are only needed in later phases.
