PYTHON ?= python3.12
VENV ?= .venv
RUN_PYTHON = $(VENV)/bin/python

.PHONY: help setup check test demo dashboard

help:
	@echo "make setup     — создать venv и установить зависимости"
	@echo "make check     — проверить документы и хеши эталонов"
	@echo "make test      — выполнить все тесты"
	@echo "make demo      — прогнать конвейер на синтетике"
	@echo "make dashboard — открыть Streamlit с результатами demo"
	@echo "Python по умолчанию: $(PYTHON); переопределение: make setup PYTHON=python3.11"

setup:
	$(PYTHON) -m venv $(VENV)
	$(RUN_PYTHON) -m pip install -r requirements.txt

check:
	$(RUN_PYTHON) tools/check_repo.py

test:
	$(RUN_PYTHON) -m pytest -q

demo:
	$(RUN_PYTHON) -m experiments.end_to_end --fixture --workdir runs/demo

dashboard:
	PT_RESULTS_DIR=runs/demo $(RUN_PYTHON) -m streamlit run app/streamlit_app.py
