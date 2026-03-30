PYTHON ?= python3

.PHONY: install prepare-data build-features eda pipeline baseline-model fetch-rentsmart phase1 phase2 test

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare-data:
	$(PYTHON) -m src.data.violations

build-features:
	$(PYTHON) -m src.data.features

eda:
	$(PYTHON) -m src.analysis.eda

pipeline:
	$(PYTHON) -m src.pipeline

baseline-model:
	$(PYTHON) -m src.modeling.baseline_model

fetch-rentsmart:
	$(PYTHON) -m src.data.context.rentsmart

phase1: prepare-data

phase2: pipeline

test:
	$(PYTHON) -m pytest -q
