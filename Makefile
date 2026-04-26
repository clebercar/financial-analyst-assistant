.PHONY: install test lint typecheck train serve eval drift smoke clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=60

lint:
	ruff check src/ tests/ evaluation/

typecheck:
	mypy src/ --ignore-missing-imports

security:
	bandit -r src/ -ll

train-lstm:
	python -m src.models.train --model lstm

train-sentiment:
	python -m src.models.train --model sentiment

serve:
	uvicorn src.serving.app:app --reload --port 8000

mlflow-ui:
	mlflow ui --port 5000

eval:
	python -m evaluation.ragas_eval

benchmark:
	python -m evaluation.benchmark_configs

drift:
	python -m src.monitoring.drift_report

download-filings:
	python -m src.data.sec_edgar

index-rag:
	python -m src.agent.rag_pipeline --reindex

smoke:
	@echo "Smoke test manual: subir docker-compose, hit /chat com curl"
	docker-compose up -d
	sleep 5
	curl -s http://localhost:8000/health | jq

clean:
	rm -rf mlruns chroma_db .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
