PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Create the venv and install the project with dev extras
	$(BIN)/pip install -e ".[dev]"

.PHONY: data
data: ## Regenerate the synthetic sales dataset
	$(BIN)/python scripts/generate_data.py

.PHONY: lint
lint: ## Run ruff on source, scripts, tests and notebooks
	$(BIN)/ruff check src scripts tests notebooks
	$(BIN)/ruff format --check src scripts tests

.PHONY: format
format: ## Auto-fix lint and formatting
	$(BIN)/ruff check --fix src scripts tests notebooks
	$(BIN)/ruff format src scripts tests

.PHONY: strip
strip: ## Strip outputs from notebooks before committing
	$(BIN)/nbstripout notebooks/*.ipynb

.PHONY: verify-notebooks
verify-notebooks: ## Fail if any notebook has committed outputs
	$(BIN)/nbstripout --verify notebooks/*.ipynb

.PHONY: test
test: ## Run the unit test suite
	$(BIN)/pytest

.PHONY: notebooks
notebooks: ## Execute every notebook and fail on error
	$(BIN)/pytest --nbmake notebooks -p no:cacheprovider

.PHONY: backtest
backtest: ## Run the rolling-origin backtest and write reports/metrics.json
	$(BIN)/python scripts/run_backtest.py

.PHONY: gate
gate: backtest ## Run the backtest and check it against eval/thresholds.yml
	$(BIN)/python scripts/check_eval_gate.py

.PHONY: check
check: lint verify-notebooks test notebooks gate ## Everything CI runs

.PHONY: clean
clean: ## Remove caches and generated reports
	rm -rf .pytest_cache .ruff_cache reports/*.json reports/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
