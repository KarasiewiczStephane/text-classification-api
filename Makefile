.PHONY: install test lint clean run docker-build docker-run docker-dev

IMAGE_NAME = text-classification-api
VERSION = 1.0.0

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models $(IMAGE_NAME):latest

docker-dev:
	docker run -it -p 8000:8000 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/models:/app/models \
		$(IMAGE_NAME):latest
