PY ?= python3
export PYTHONPATH := src:../packages/oeq-core/src

.PHONY: setup test reproduce-sample report dashboard check

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

reproduce-sample:
	$(PY) -m polymarket_factors.cli reproduce-sample

report:
	$(PY) src/polymarket_factors/make_report_figs_en.py

dashboard:
	$(PY) -m uvicorn polymarket_factors.app:app --port 8000

check: test reproduce-sample
