# Sentiment Analysis Microservice

A complete end-to-end microservice for binary sentiment analysis using Hugging Face Transformers, Flask, and React.

## 🚀 Features

- **Backend API**: Flask-based REST API with sentiment analysis using DistilBERT
- **Frontend**: React-based web interface with real-time sentiment prediction
- **Fine-tuning**: Standalone CLI script for model fine-tuning
- **Containerization**: Docker and Docker Compose for easy deployment
- **Model Management**: Automatic loading of fine-tuned models
- **Health Checks**: Built-in health monitoring for all services

## 📁 Project Structure

```
sentiment-analysis/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Backend container
│   └── Dockerfile.gpu        # GPU-enabled backend
├── frontend/
│   ├── src/
│   │   ├── App.js            # React main component
│   │   ├── App.css           # Styling
│   │   └── index.js          # React entry point
│   ├── public/
│   │   └── index.html        # HTML template
│   ├── package.json          # Node.js dependencies
│   ├── Dockerfile            # Frontend container
│   └── nginx.conf            # Nginx configuration
├── finetune.py               # Fine-tuning script
├── data.jsonl                # Sample training data
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # This file
```

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- Docker & Docker Compose (for containerized deployment)

### Local Development

#### 1. Backend Setup

```bash
# Create and navigate to project directory
mkdir sentiment-analysis
cd sentiment-analysis

# Create backend directory
mkdir backend
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
```

The backend will be available at `http://localhost:8000`

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

#### 3. Fine-tuning (Optional)

```bash
# Navigate to project root
cd ..

# Create sample training data
python finetune.py --create_sample_data --data data.jsonl

# Run fine-tuning
python finetune.py --data data.jsonl --epochs 3 --lr 3e-5
```

### Docker Deployment

#### Quick Start

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

#### GPU Support (Optional)

```bash
# Start with GPU support
docker-compose --profile gpu up --build
```

## 📚 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "device": "cpu"
}
```

#### `POST /predict`
Sentiment prediction endpoint

**Request:**
```json
{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "label": "positive",
  "score": 0.9245
}
```

#### `GET /model/info`
Get model information
```json
{
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "device": "cpu",
  "model_path": "./model",
  "is_fine_tuned": false
}
```

#### `POST /model/reload`
Reload model (useful after fine-tuning)
```json
{
  "message": "Model reloaded successfully"
}
```

## 🔧 Configuration

### Environment Variables

#### Backend
- `FLASK_ENV`: Flask environment (development/production)
- `PYTHONUNBUFFERED`: Python output buffering

#### Frontend
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8000)

### Fine-tuning Parameters

The fine-tuning script accepts the following parameters:

```bash
python finetune.py --help

Options:
  --data TEXT         Path to training data file (JSONL format) [required]
  --epochs INTEGER    Number of training epochs [default: 3]
  --lr FLOAT         Learning rate [default: 3e-5]
  --batch_size INTEGER Batch size [default: 16]
  --model TEXT       Base model to fine-tune [default: distilbert-base-uncased-finetuned-sst-2-english]
  --create_sample_data Create sample training data file
```

### Training Data Format

The training data should be in JSONL format (one JSON object per line):

```jsonl
{"text": "I love this product!", "label": "positive"}
{"text": "This is terrible.", "label": "negative"}
```

## 🏗️ Design Decisions

### Architecture
- **Microservice Pattern**: Separate backend and frontend services
- **REST API**: Simple and widely supported
- **Containerization**: Docker for consistent deployment across environments

### Technology Stack
- **Backend**: Flask (lightweight, simple to deploy)
- **ML Framework**: PyTorch + Transformers (industry standard)
- **Frontend**: React (component-based, responsive)
- **Model**: DistilBERT (fast inference, good performance)

### Key Features
- **Automatic Model Loading**: Checks for fine-tuned models on startup
- **Gradient Clipping**: Prevents exploding gradients during training
- **Learning Rate Scheduling**: Improves training stability
- **Deterministic Training**: Reproducible results with fixed seeds
- **Health Monitoring**: Built-in health checks for all services

## ⚡ Performance

### Inference Times (Approximate)
- **CPU**: 100-200ms per request
- **GPU**: 20-50ms per request

### Memory Usage
- **Model Loading**: ~500MB RAM
- **Backend Service**: ~1GB RAM
- **Frontend Service**: ~100MB RAM

### Training Times (Sample Dataset)
- **CPU**: ~2-3 minutes for 30 samples, 3 epochs
- **GPU**: ~30-60 seconds for 30 samples, 3 epochs

## 🐳 Docker Configuration

### Services

1. **Backend** (Port 8000)
   - Flask API server
   - Model inference
   - Health checks

2. **Frontend** (Port 3000)
   - React application
   - Nginx reverse proxy
   - Static file serving

3. **Backend-GPU** (Port 8001, Optional)
   - GPU-enabled backend
   - CUDA support
   - Requires nvidia-docker

### Volumes
- `./model:/app/model` - Shared model directory

### Networks
- Default bridge network for service communication

## 🔍 Monitoring & Debugging

### Health Checks
All services include health check endpoints:
- Backend: `http://localhost:8000/health`
- Frontend: `http://localhost:3000/health`

### Logs
```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Debug Mode
```bash
# Start with debug logging
FLASK_DEBUG=1 python backend/app.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model directory permissions
   - Verify model files are complete
   - Check available memory

2. **API Connection Issues**
   - Verify backend is running on port