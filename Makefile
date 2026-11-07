PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install dev test test-hw lint type check ci clean bench report

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q -m "not hw"

test-hw:
	$(PYTHON) -m pytest -q -m hw

lint:
	$(PYTHON) -m ruff check src/mathcore src/intent.py scripts/measure_stt.py scripts/measure_wakeword.py scripts/fit_thresholds.py scripts/report.py

type:
	$(PYTHON) -m mypy src/mathcore

check:
	@for file in tests/check_*.py; do \
		$(PYTHON) "$$file" || exit 1; \
	done

ci: lint type test

clean:
	rm -rf .pytest_cache .coverage htmlcov

bench:
	$(PYTHON) scripts/measure_stt.py --self-test --json /tmp/jarvis-stt.json

report:
	$(PYTHON) scripts/report.py --self-test --json /tmp/jarvis-report.json --out /tmp/jarvis-report
