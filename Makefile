SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: test typecheck format profile

test:
	uv run --active pytest

typecheck:
	uv run --active pyright .

format:
	uv run --active ruff format .

profile:
	prof_file=$$(mktemp /tmp/plutil.profile.XXXXXX) && uv run --active python -m cProfile src/plutil/__main__.py > "$$prof_file" && echo "$$prof_file"
