SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: test typecheck format

test:
	uv run --active pytest -p varspec.pytest_plugin --varspec-content --import-mode=importlib $(TEST_PATHS) $(CONTENT_TEST_PATHS) $(PYTEST_ARGS)

typecheck:
	uv run --active pyright .

format:
	uv run --active ruff format .
