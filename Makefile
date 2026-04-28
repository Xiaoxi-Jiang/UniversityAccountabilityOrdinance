PYTHON ?= python3

.PHONY: install prepare-data build-features eda interactive-viz pipeline baseline-model improved-model fetch-rentsmart phase1 phase2 test

install:
	$(PYTHON) -m pip install -r requirements.txt

fetch-rentsmart:
	$(PYTHON) -m src.data.context.rentsmart

prepare-data:
	$(PYTHON) -m src.data.violations

build-features:
	$(PYTHON) -m src.data.features

eda:
	$(PYTHON) -m src.analysis.eda

interactive-viz:
	$(PYTHON) -m src.viz.interactive_visualizations

pipeline:
	$(PYTHON) -m src.pipeline

baseline-model:
	$(PYTHON) -m src.modeling.baseline_model

improved-model:
	$(PYTHON) -m src.modeling.improved_model

phase1: prepare-data

phase2: pipeline

test:
	$(PYTHON) -m pytest -q
