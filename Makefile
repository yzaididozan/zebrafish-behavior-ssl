.PHONY: setup test ci baseline help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make setup     Create .venv and install project + dev dependencies"
	@echo "  make test      Run the local pytest suite"
	@echo "  make ci        Run pytest in the current environment"
	@echo "  make baseline  Reproduce DS-005 TRAIN/VALIDATION baseline clustering selection"

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,viz]"

test:
	$(PYTHON) -m pytest -q

ci:
	python -m pytest -q

baseline:
	$(PYTHON) src/discovery/baseline_clustering.py select
