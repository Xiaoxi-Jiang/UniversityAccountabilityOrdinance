.PHONY: install phase1 test

install:
	python -m pip install -r requirements.txt

phase1:
	python -m src.data.phase1_pipeline

test:
	pytest -q
