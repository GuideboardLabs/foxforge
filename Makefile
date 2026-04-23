.PHONY: lint smoke test ui-smoke check

PYTHON_BIN := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH_SRC := PYTHONPATH=SourceCode

lint:
	$(PYTHON_BIN) -m compileall SourceCode tests smoke_test.py run_integration_tests.py tools

smoke:
	$(PYTHONPATH_SRC) $(PYTHON_BIN) smoke_test.py

test:
	$(PYTHONPATH_SRC) $(PYTHON_BIN) run_integration_tests.py

ui-smoke:
	$(PYTHONPATH_SRC) $(PYTHON_BIN) tools/ui_phase_smoke.py

check: lint smoke test ui-smoke
	$(PYTHONPATH_SRC) $(PYTHON_BIN) tools/repo_health_check.py
