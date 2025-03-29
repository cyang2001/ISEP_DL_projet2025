# Sign Language Recognition System

## Project Overview

This project aims to develop a deep learning-based video sign language recognition system capable of identifying sign language gestures from video sequences. The system is implemented using PyTorch and supports multi-platform development environments.

## Features

- Video preprocessing and feature extraction
- Transformer-based sequence modeling
- Skeleton point extraction and integration
- Multi-platform compatibility (Windows/Linux/macOS)
- Comprehensive evaluation and visualization tools

## Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU training)
- See `requirements.txt` for complete dependencies

## Quick Start

### Using Docker (Recommended)

```bash
# Build Docker image
docker build -t sign_language_recognition .

# Run container
docker run -it --gpus all -v $(pwd):/workspace sign_language_recognition
```

### Manual Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sign_language_recognition.git
cd sign_language_recognition

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

## Usage

### Data Preprocessing

```bash
python scripts/preprocess.py --config configs/default.yaml
```

### Training Models

```bash
python scripts/train.py --config configs/experiments/transformer.yaml
```

### Evaluating Models

```bash
python scripts/evaluate.py --model_path saved_models/model.pt --data_path data/processed/test
```

## Project Structure

See [Architecture Documentation](docs/architecture.md) for details.

## Development Guide

For contributors, please refer to our [Development Guide](docs/api/development_guide.md).

## License

[MIT License](LICENSE)

## Contact

For questions, please contact the project supervisor. 