SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: test typecheck format

test:
	uv run --active pytest

typecheck:
	uv run --active pyright .

format:
	uv run --active ruff format .
