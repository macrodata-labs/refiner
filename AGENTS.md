# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `src/refiner/`.
- Main domains are split by concern: `readers/` (CSV, JSONL, Parquet input), `io/` (datafile/datafolder/fileset abstractions), `ledger/` (state + backend adapters), `processors/`, and `pipeline.py`/`worker.py` for orchestration.
- Tests mirror source modules under `tests/` (for example, `tests/readers/test_csv_reader.py` maps to `src/refiner/readers/csv.py`).
- Project metadata and tooling are configured in `pyproject.toml`; hook automation is in `.pre-commit-config.yaml`.

## Build, Test, and Development Commands
- `uv sync --dev`: install runtime + development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the full test suite (`tests/`).
- `uv run pytest tests/readers/test_parquet_reader.py`: run a focused test module during iteration.
- `uv run ruff check --force-exclude --fix .`: lint and apply safe fixes.
- `uv run ruff format --force-exclude .`: format Python files.
- `uv run ty check`: run static type checks.
- `uv run pre-commit run --all-files`: execute the same checks used by local hooks.

## Coding Style & Naming Conventions
- Target Python `>=3.10`; use 4-space indentation and type annotations on public APIs.
- File and module names use `snake_case`; classes use `PascalCase`; functions/variables use `snake_case`.
- Keep reader/ledger implementations modular by backend or format (`csv.py`, `redis.py`, `fs.py`).
- Use Ruff as the formatter/linter source of truth; do not hand-format against Ruff output.
- Avoid copy-paste duplication; extract shared logic into focused helpers/modules.
- Avoid oversized monolith files; split by responsibility when files grow beyond clear maintainability.
- Do not add blanket `try/except` blocks; handle exceptions only where there is a specific recovery or translation plan.

## Testing Guidelines
- Framework: `pytest` with tests discovered from `tests/` (`[tool.pytest.ini_options]`).
- Name tests as `test_*.py`, and functions as `test_<behavior>()`.
- Add or update tests alongside code changes in the matching test area (reader, io, ledger, etc.).
- Prefer small, deterministic unit tests over broad integration tests unless cross-module behavior changed.

## Commit & Pull Request Guidelines
- Existing history favors short, imperative, lowercase commit subjects (for example, `renaming to refiner`, `added readers...`).
- Keep subject lines concise and specific; group related changes per commit.
- PRs should include: purpose, key design decisions, test evidence (`uv run pytest` output summary), and any follow-up work.
- Link relevant issues/tasks and include before/after behavior notes when changing pipeline or ledger semantics.

## Agent Collaboration Behavior
- Keep feedback factual, direct, and technically grounded; avoid praise-only or agreeable language that hides risk.
- Do not use sycophantic phrasing or validation preambles (for example, “you’re right”, “great point”, or similar filler); respond directly with technical content.
- When discussing architecture or proposed changes, challenge weak assumptions and surface tradeoffs explicitly.
- If a proposal increases complexity, risk, or maintenance cost, push back with concrete alternatives and rationale.
- Prioritize correctness and long-term maintainability over politeness-driven agreement.
- Never “cheat” to make checks pass: do not remove or weaken tests, disable quality gates, or bypass failures without explicit approval.
- Do not preserve backwards compatibility by default; when architecture or API cleanup is requested, remove legacy paths unless the user explicitly asks to keep compatibility.
- For architecture and execution-model decisions, explicitly compare approaches with Spark, Beam/Dataflow, Daft, Hugging Face Datasets, and Ray/Ray Data, then justify deviations for this codebase.

## Documentation Discipline
- Keep project docs in Mintlify-style Markdown with frontmatter (`title`, `description`) under `docs/`.
- Any new feature, execution block, architectural change, or user-visible behavior change must include corresponding doc updates in the same change set.
- Write docs user-first: explain how to use the API/workflow and expected behavior before discussing internals.
- Include internal implementation notes only at the end of each doc in a short `## Internal Notes` section.
- In docs, import from the package top level (`import refiner as mdr`) instead of module-level imports.
