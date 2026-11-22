PY ?= python
.DEFAULT_GOAL := help

.PHONY: help install test lint format demo clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the suite (offline, no API key)
	$(PY) -m pytest

lint:  ## Static checks
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Apply formatting and safe fixes
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

demo:  ## Show a refusal and a routing decision
	$(PY) examples/demo.py

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
