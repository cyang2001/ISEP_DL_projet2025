# Development Guide

## Introduction

This document provides guidelines and best practices for developers working on the Sign Language Recognition project. It covers development workflow, code standards, documentation, and testing.

## Development Environment

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sign_language_recognition.git
   cd sign_language_recognition
   ```

2. **Set up a virtual environment**:
   ```bash
   # Using venv (recommended)
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   
   # Using conda
   conda create -n sign_language python=3.8
   conda activate sign_language
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the project in development mode
   ```

### Docker Environment (Alternative)

For consistency across different platforms, we recommend using Docker:

```bash
# Build the Docker image
docker build -t sign_language_recognition .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace sign_language_recognition
```

## Project Structure

```
sign_language_recognition/
├── src/                    # Source code
│   ├── utils/              # Utility functions
│   ├── data/               # Data processing
│   ├── models/             # Model definitions
│   ├── trainers/           # Training code
│   └── inference/          # Inference code
├── configs/                # Configuration files
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── scripts/                # Training & evaluation scripts
├── notebooks/              # Research notebooks
└── data/                   # Data directory (gitignored)
```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code and tests**:
   - Follow the code style guidelines
   - Add unit tests for new functionality
   - Update documentation as needed

3. **Run tests locally**:
   ```bash
   pytest tests/
   ```

4. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add feature xyz"
   ```

5. **Push to GitHub and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Code review and merge**:
   - The project supervisor will review code changes
   - Address any review comments
   - Changes will be merged after approval

## Code Style Guidelines

### Python Conventions

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (not tabs)
- Maximum line length of 88 characters (following Black defaults)
- Use meaningful variable names

### Docstrings

All modules, classes, and functions must include Google-style docstrings in French:

```python
def extract_features(video_path, num_frames=64):
    """
    Extrait les caractéristiques d'une vidéo.
    
    Cette fonction charge une vidéo, extrait un nombre fixe de cadres,
    et calcule les caractéristiques visuelles.
    
    Args:
        video_path (str): Chemin vers le fichier vidéo.
        num_frames (int, optional): Nombre de cadres à extraire. Default: 64.
        
    Returns:
        np.ndarray: Tableau de caractéristiques de forme [num_frames, feature_dim].
        
    Raises:
        FileNotFoundError: Si le fichier vidéo n'existe pas.
    """
    # Implementation...
```

### Imports

Organize imports in this order:
1. Standard library imports
2. Related third-party imports
3. Local application imports

Separate each group with a blank line:

```python
import os
import sys
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import get_logger
from src.data.preprocessing import preprocess_video
```

## Pull Request Process

1. **Create focused PRs**: Each PR should address a single feature or bug fix
2. **Include tests**: Add relevant tests for your changes
3. **Update documentation**: Update any documentation that is affected by your changes
4. **Ensure all checks pass**: PRs must pass all automated tests
5. **Code review**: Wait for code review and address any comments

## Testing Guidelines

- Write unit tests for all new functions and classes
- Use pytest for testing
- Place tests in the `tests/` directory, mirroring the structure of `src/`
- Aim for at least 80% code coverage

Example test:

```python
def test_extract_features():
    # Arrange
    video_path = "tests/test_data/sample_video.mp4"
    num_frames = 32
    
    # Act
    features = extract_features(video_path, num_frames)
    
    # Assert
    assert features.shape[0] == num_frames
    assert features.shape[1] > 0  # Should have some features
```

## Logging

Use the project's logger for all logging:

```python
from src.utils import get_logger

logger = get_logger(__name__)

def process_video(video_path):
    logger.info(f"Processing video: {video_path}")
    # Implementation...
    logger.debug("Extracted frames with shape: %s", frames.shape)
```

## Configuration Management

- Store all configurable parameters in YAML files under `configs/`
- Access configuration using the provided utilities
- Don't hardcode parameters that should be configurable

## Experiment Tracking

Use Weights & Biases (wandb) for experiment tracking:

```python
import wandb

# Initialize wandb
wandb.init(project="sign_language_recognition", config=config)

# Log metrics during training
wandb.log({"train_loss": loss, "val_accuracy": accuracy})

# Log artifacts
wandb.save("model.pt")
```

## Multi-Platform Considerations

- Use `os.path` for file path operations to ensure cross-platform compatibility
- Avoid platform-specific libraries when possible
- Test code on multiple platforms when feasible
- Use Docker to ensure consistent environments

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Weights & Biases Documentation](https://docs.wandb.ai/) 