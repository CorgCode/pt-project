.PHONY: install data pipeline serve all test docker clean

PY ?= python3
DATA ?= data
ARTIFACTS ?= artifacts
WINDOWS ?= 5

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m hostembed.datagen --out $(DATA) --windows $(WINDOWS)

pipeline:
	$(PY) -m hostembed.pipeline --data $(DATA) --artifacts $(ARTIFACTS)

serve:
	$(PY) -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload

all: data pipeline serve

test:
	$(PY) -m pytest -q

docker:
	docker compose up --build

clean:
	rm -rf $(DATA) $(ARTIFACTS) **/__pycache__ .pytest_cache
