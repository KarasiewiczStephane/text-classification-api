# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for local deployment.

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_version": "distilbert_v1",
  "model_loaded": true
}
```

### Single Prediction

```
POST /predict
```

**Request Body:**
```json
{
  "text": "Apple announces new iPhone with advanced features"
}
```

**Response:**
```json
{
  "label": "Sci/Tech",
  "confidence": 0.94,
  "probabilities": {
    "World": 0.02,
    "Sports": 0.01,
    "Business": 0.03,
    "Sci/Tech": 0.94
  },
  "model_version": "distilbert_v1",
  "uncertain": false
}
```

### Batch Prediction

```
POST /predict/batch
```

**Request Body:**
```json
{
  "texts": [
    "Stock market rises amid economic recovery",
    "Team wins championship in overtime"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {"label": "Business", "confidence": 0.91, ...},
    {"label": "Sports", "confidence": 0.96, ...}
  ],
  "total": 2,
  "model_version": "distilbert_v1"
}
```

### List Models

```
GET /models
```

### Switch Model

```
POST /models/switch?version=bert_v1
```

### A/B Test Configuration

```
PUT /ab-test/config
```

**Request Body:**
```json
{
  "model_a": "distilbert_v1",
  "model_b": "bert_v1",
  "split_ratio": 0.8
}
```

### A/B Test Results

```
GET /ab-test/results
```

## Error Codes

| Code | Description |
|------|-------------|
| 422 | Validation error (invalid input) |
| 404 | Model version not found |
| 503 | Model not loaded |
