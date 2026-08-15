SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: test typecheck format check-format profile ci-dryrun

test:
	uv run --active pytest

typecheck:
	uv run --active pyright .

format:
	uv run --active ruff format .

check-format:
	uv run --active ruff format --check .

ci-dryrun: test typecheck check-format