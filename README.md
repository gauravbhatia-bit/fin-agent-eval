# fin-agent-eval

A local-first evaluation harness for a finance-education chatbot using fictional demo data. This is a prototype for evaluating AI agent reliability, not a financial product. It does not provide investment advice and is not affiliated with any broker.

## Status

Phases 1-7 are implemented. Phase 3 adds the record/replay/live Gemini client, hybrid BM25/embedding retrieval, guarded LangGraph agent, and `POST /chat`. Phase 4 adds reviewable golden cases, metrics, judge auditing, resumable evaluation runs, a quality gate, and report comparison. Phase 5 adds a Streamlit UI backed only by the HTTP API. Phase 6 packages the API and UI for local Docker use. Phase 7 adds GitHub Actions checks and Dependabot. API keys are not committed; live Gemini requests may use quota.

## Requirements

- Python 3.12
- Git
- Streamlit (installed by project setup)
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

Run the Streamlit UI in a second terminal:

```powershell
py -3.12 tasks.py run-ui
```

Open `http://localhost:8501`. The UI reads `BACKEND_URL` (default
`http://localhost:8000`) and retries the health check for up to 90 seconds to
handle a backend cold start. When no evaluation has been run, the dashboard
shows clearly marked sample data from `reports/sample_latest.json`.

## Fictional pack and tools

The `packs/eu_us_etf/` files contain invented DemoBroker fees, instruments, and dated currency rates. They are for software tests only and are not current real-world data. Its 18 knowledge documents are unverified drafts; educational descriptions marked `TODO(verify)` need owner review before being treated as verified.

The tool registry in `app/tools/registry.py` exposes fee, hypothetical-growth, currency-conversion, instrument-lookup, position-size, knowledge-search, and simulated-escalation tools. Each request is validated with Pydantic. Invalid arguments return an explicit error. `POST /fees` runs the fictional fee calculator locally without calling Gemini. `POST /chat` applies policy rules before LLM calls, uses a manually bounded tool loop, and validates document citations. In-memory sessions retain up to six turns for 30 minutes and are lost on restart. The hybrid retriever caches document embeddings locally, separated by embedding model.

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
generation or embedding cassette is missing. The CI replay evaluation runs only
when matching cassettes are checked in; until then CI runs deterministic API and
tool smoke tests and explicitly does not claim model-quality evaluation.
`--resume` keeps completed
case/run pairs in the output report. Baseline targets are placeholders, not
measured scores. Manually reviewed judge labels belong in
`evals/human_labels.jsonl`; the audit reports unavailable until labels are
provided. `GET /eval/latest` returns a summary of `reports/latest.json` when it
exists, or the clearly labelled sample report when no run report is present.

## Streamlit UI (Phase 5)

The UI has Chat, Evaluation, and About and limits tabs. The Chat tab includes a
local fee calculator using `POST /fees`; it works without a Gemini key, quota,
or billing. Ordinary chat and model-based evaluation still depend on Gemini;
when its quota is exhausted, chat reports that AI is temporarily unavailable
without affecting the local calculator. The UI communicates with the backend
only over HTTP using `BACKEND_URL`; it does not import application modules. Chat
history remains in browser session state. The dashboard shows
the sample report until a run report is available. Sample values and baseline
thresholds are illustrative, not measured results. Chat requests allow up to
180 seconds for a model response; connection failures and response timeouts are
shown separately. After a response timeout, check the API logs before retrying,
because the backend request may still be processing. The Gemini SDK request
itself is capped at 30 seconds so provider/network stalls do not leave API
workers waiting indefinitely.

## Docker (Phase 6)

Docker Desktop is required. Copy `.env.example` to `.env` and set a valid
Gemini API key and model IDs before using ordinary chat questions. The Fee
calculator works locally without those settings. Without Gemini settings,
`/health`, `/fees`, and deterministic policy refusals work, but model questions
return a configuration error and cannot be evaluated as answer quality.

Build and run the API and Streamlit UI together:

```powershell
docker compose up --build
```

Open `http://localhost:8501`; the API is available at
`http://localhost:8000/health`. The API image runs as a non-root user and
defaults to port 10000; Compose sets its internal port to 8000 for local use.
If those host ports are already in use, set `API_HOST_PORT` and `UI_HOST_PORT`
in `.env` to available ports.
Stop the services with `Ctrl+C`, then run `docker compose down`.

To build and smoke-test only the API image:

```powershell
docker build --build-arg GIT_SHA=local -t fin-agent-eval:local .
docker run --rm -p 8000:10000 -e LLM_MODE=replay -e GIT_SHA=local fin-agent-eval:local
```

In another terminal, check `http://localhost:8000/health`; it should report
the configured Git SHA. The image excludes the Streamlit source; Compose
mounts the local `ui/` directory into a second container using the same
dependency image.

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for phase scope, acceptance criteria, deployment requirements, and known limitations.

## GitHub setup

The CI workflow runs lint, tests, offline smoke checks (or replay evaluation
when cassettes are available), and a Docker health/fee-calculation check on
pull requests and pushes to `main`. Terraform validation activates when
Terraform files are added in Phase 8. Dependabot checks Python dependencies,
GitHub Actions, and Docker weekly; Terraform updates can be enabled when its
configuration is added.

To protect `main`, open **Settings → Branches → Add branch protection rule**,
set the branch name pattern to `main`, and require pull requests plus these
status checks: `lint-test`, `eval-smoke`, `docker-build`, and
`terraform-validate`. Keep force pushes and branch deletion disabled. The
Terraform check currently reports that validation is deferred because this
repository does not yet contain Terraform configuration. Add the Terraform
plan/comment workflow and required secrets only as part of Phase 8.
