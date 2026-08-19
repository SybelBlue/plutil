SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: venv deps clean test typecheck format check-format check-prairielearn-dependencies profile ci-dryrun

venv:
	uv venv --clear

deps: venv
	uv sync --locked --active

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

test:
	uv run --active pytest

typecheck:
	uv run --active pyright .

format:
	uv run --active ruff format .

check-format:
	uv run --active ruff format --check .

check-prairielearn-dependencies:
	uv run --active python scripts/check-prairielearn-dependencies.py

ci-dryrun:
	@./scripts/ci-dryrun.sh \
		"$(MAKE) test" \
		"$(MAKE) typecheck" \
		"$(MAKE) check-format" \
		"$(MAKE) check-prairielearn-dependencies"
