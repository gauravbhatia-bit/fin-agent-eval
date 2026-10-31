# Coding assistant instructions

- Follow `ROADMAP.md` one phase at a time and stop at each phase's acceptance checks.
- Never invent package versions, API signatures, model names, Terraform resources, or provider arguments. Verify documentation first; ask if requirements conflict.
- Pin dependencies and keep the lock file current. Add tests with each feature.
- Use type hints and Pydantic models for API and file schemas.
- Keep secrets out of source control; `.env` is local-only.
- Never alter evaluation baselines or thresholds unless explicitly requested.
- Use only the fictional broker DemoBroker and clearly label its data as fictional.
- Do not add trading, real brokerage access, personalized investment advice, predictions, or real customer data.
- Write plain, direct English and keep changes scoped to the current phase.
