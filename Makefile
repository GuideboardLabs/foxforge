.PHONY: lint smoke test ui-smoke check

lint:
	python -m compileall SourceCode tests smoke_test.py run_integration_tests.py tools

smoke:
	python smoke_test.py

test:
	python run_integration_tests.py

ui-smoke:
	python tools/ui_phase_smoke.py

check: lint smoke test ui-smoke
	python tools/repo_health_check.py
