SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: venv deps clean test typecheck format check-format check-prairielearn-dependencies profile ci-dryrun docs

venv:
	uv venv --clear

deps: venv
	uv sync --locked --active

clean:
	rm -rf .venv .pytest_cache .ruff_cache docs/_build docs/_autosummary
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
	uv run --active pl-vendor check prairielearn python

docs:
	rm -rf docs/_build docs/_autosummary
	uv run --active --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

ci-dryrun:
	@./scripts/ci-dryrun.sh \
		"$(MAKE) test" \
		"$(MAKE) typecheck" \
		"$(MAKE) check-format" \
		"$(MAKE) check-prairielearn-dependencies"
