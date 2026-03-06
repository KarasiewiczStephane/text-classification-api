# Text Classification API

> Multi-class text classification using fine-tuned DistilBERT and BERT models with A/B testing infrastructure.

## Overview

A FastAPI-based service for document classification using the AG News dataset (4 classes: World, Sports, Business, Sci/Tech). Features model versioning, A/B testing for comparing model performance, and a Streamlit dashboard for monitoring.

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

+--------------------------+
|   Streamlit Dashboard    |
|  A/B metrics, latency,   |
|  confusion matrix, dist  |
+--------------------------+
```

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/KarasiewiczStephane/text-classification-api.git
cd text-classification-api
make install
```

### 2. Download data

The project uses the AG News dataset from Hugging Face. The downloader fetches it automatically during training, but you can also download it explicitly in Python:

```python
from src.data.downloader import download_ag_news, create_sample_data

dataset = download_ag_news()          # Downloads to data/raw/
create_sample_data(dataset)           # Creates data/sample/sample.json
```

### 3. Train a model

```bash
# Train DistilBERT (default, faster)
python -m src.train --model distilbert

# Train BERT (higher accuracy)
python -m src.train --model bert
```

Trained models are saved to `models/`.

### 4. Run the API

```bash
# Development with hot-reload
make run
# Serves at http://localhost:8000

# Production (Docker)
make docker-build && make docker-run
```

### 5. Launch the dashboard

```bash
make dashboard
# Opens Streamlit at http://localhost:8501
```

The dashboard displays A/B test metrics, model comparison charts, prediction distribution, latency trends, and a confusion matrix using synthetic demo data.

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
│   ├── api/           # FastAPI app, routes, schemas, middleware
│   ├── data/          # Data downloading, preprocessing, analysis
│   ├── models/        # Training, evaluation, model registry
│   ├── benchmarks/    # Performance benchmarking
│   ├── dashboard/     # Streamlit monitoring dashboard
│   └── utils/         # Config loader, logging utilities
├── tests/             # Unit and integration tests
├── configs/           # YAML configuration files
├── models/            # Trained model artifacts
├── data/              # Raw and sample datasets
├── docs/              # Documentation and benchmarks
├── .github/workflows/ # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Makefile Targets

| Target | Command |
|--------|---------|
| `make install` | Install Python dependencies |
| `make run` | Start API server on port 8000 with hot-reload |
| `make dashboard` | Launch Streamlit dashboard on port 8501 |
| `make test` | Run tests with coverage |
| `make lint` | Lint and format with Ruff |
| `make clean` | Remove `__pycache__` and `.pyc` files |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container on port 8000 |

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
