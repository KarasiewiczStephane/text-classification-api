# Text Classification API

> Multi-class text classification using fine-tuned DistilBERT and BERT models with A/B testing infrastructure.

## Overview

A FastAPI-based service for document classification using the AG News dataset (4 classes: World, Sports, Business, Sci/Tech). Features model versioning, A/B testing for comparing model performance, and comprehensive benchmarks.

## Architecture

```
+-------------------------------------------------------------+
|                        FastAPI                               |
|  +-----------+  +-----------+  +---------------------+      |
|  | /predict  |  | /models   |  |    /ab-test         |      |
|  +-----+-----+  +-----+-----+  +----------+----------+      |
|        |               |                   |                 |
|        v               v                   v                 |
|  +-----------------------------------------------------+    |
|  |                   A/B Router                         |    |
|  |         (traffic splitting, metrics tracking)        |    |
|  +------------------------+----------------------------+     |
|                           |                                  |
|  +------------------------+------------------------+         |
|  |              Model Registry                     |         |
|  |    +---------------+  +---------------+         |         |
|  |    |  DistilBERT   |  |     BERT      |         |         |
|  |    |    v1, v2...  |  |   v1, v2...   |         |         |
|  |    +---------------+  +---------------+         |         |
|  +-------------------------------------------------+         |
+-------------------------------------------------------------+
```

## Quick Start

### Installation

```bash
git clone https://github.com/KarasiewiczStephane/text-classification-api.git
cd text-classification-api
pip install -r requirements.txt
```

### Training Models

```bash
# Train DistilBERT
python -m src.train --model distilbert

# Train BERT
python -m src.train --model bert
```

### Running the API

```bash
# Development
make run

# Production (Docker)
make docker-build && make docker-run
```

### API Usage

```python
import httpx

# Single prediction
response = httpx.post(
    "http://localhost:8000/predict",
    json={"text": "Apple announces new iPhone with advanced features"}
)
print(response.json())
# {"label": "Sci/Tech", "confidence": 0.94, "probabilities": {...}}

# Batch prediction
response = httpx.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Stock market rises...", "Team wins championship..."]}
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single text classification |
| `/predict/batch` | POST | Batch classification (up to 50) |
| `/health` | GET | Health check with model info |
| `/models` | GET | List available models |
| `/models/switch` | POST | Switch active model |
| `/models/active` | GET | Get active model info |
| `/ab-test/config` | GET/PUT | A/B test configuration |
| `/ab-test/results` | GET | A/B test metrics |
| `/ab-test/reset` | POST | Reset A/B metrics |
| `/docs` | GET | OpenAPI documentation |

## Benchmark Results

| Model | Accuracy | p50 Latency | p99 Latency | Memory |
|-------|----------|-------------|-------------|--------|
| DistilBERT | 92.1% | 12ms | 25ms | 265MB |
| BERT | 92.8% | 24ms | 48ms | 420MB |

DistilBERT offers 2x faster inference with minimal accuracy tradeoff.

## Project Structure

```
text-classification-api/
├── src/
│   ├── api/           # FastAPI routes, schemas, middleware
│   ├── data/          # Data loading, preprocessing, analysis
│   ├── models/        # Training, evaluation, registry
│   ├── benchmarks/    # Performance benchmarking
│   └── utils/         # Config, logging utilities
├── tests/             # Unit and integration tests
├── configs/           # YAML configuration files
├── models/            # Trained model artifacts
├── docs/              # Documentation and benchmarks
├── .github/workflows/ # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Development

### Testing

```bash
# Run all tests
make test

# With coverage report
pytest tests/ -v --cov=src --cov-report=html
```

### Linting

```bash
make lint
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## License

MIT
