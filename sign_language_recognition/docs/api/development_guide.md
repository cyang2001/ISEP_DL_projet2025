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
   # Using conda (recommended)
   conda create -n sign_language python=3.10
   conda activate sign_language
   
   # Alternative: using venv
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install the project in development mode
   ```

4. **For GPU support**:
   Install the appropriate CUDA toolkit version (CUDA 12.4 recommended) according to your PyTorch version.

### Docker Environment (Optional)

Docker environment is primarily intended for inference/deployment rather than development. For development, we recommend using a conda environment as described above.

If you still need Docker for your workflow:

```bash
docker login # First time of using docker
# Build the Docker image
docker build -t sign_language_recognition .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace sign_language_recognition
```

## Project Structure

```
sign_language_recognition/
├── run.py                  # Main entry point
├── src/                    # Source code
│   ├── utils/              # Utility functions
│   ├── data/               # Data processing modules
│   │   ├── data_preprocessor.py  # Class implementations
│   │   └── ...
│   ├── data_preprocessing.py     # Main entry module for preprocessing
│   ├── models/             # Model definitions
│   └── ...
├── configs/                # Configuration files
│   ├── config.yaml         # Main configuration file
│   ├── mode/               # Running mode configurations
│   │   ├── default.yaml    # Default mode settings
│   │   └── test.yaml       # Test mode settings
│   ├── model/              # Model-specific configurations
│   │   ├── data_preprocessing.yaml
│   │   └── ...
│   └── wandb/              # WandB configurations
│       └── wandb.yaml
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── scripts/                # Helper scripts
├── notebooks/              # Research notebooks
└── data/                   # Data directory (gitignored)
    ├── raw/                # Raw video files
    │   └── WLASL/          # WLASL dataset videos
    ├── processed/          # Processed data files
    └── WLASL_v0.3.json     # WLASL dataset annotations
```

## WLASL Dataset Integration

### Overview

The project supports processing videos from the [WLASL dataset](https://github.com/dxli94/WLASL) (Word-Level American Sign Language). The dataset integration includes:

- Video preprocessing with configurable parameters
- Support for the WLASL JSON annotation format
- Handling of missing videos with detailed reporting
- Dataset statistics generation

### Configuration

WLASL-specific configuration is located in three places:

1. **Main configuration file** (`configs/config.yaml`):
   ```yaml
   data:
     dataset: "wlasl"
     raw_dir: "data/raw/WLASL"
     processed_dir: "data/processed"
     wlasl_json_path: "data/WLASL_v0.3.json"
     missing_videos_log: "data/missing_videos.txt"
   ```

2. **Mode configuration** (`configs/mode/default.yaml`):
   ```yaml
   mode:
     # WLASL specific settings
     find_missing: false  # Whether to only check for missing files
   ```

3. **Preprocessing configuration** (`configs/model/data_preprocessing.yaml`):
   ```yaml
   data_preprocessing:
     # WLASL specific parameters
     wlasl:
       subset_size: 2000
       only_split_videos: false
       train_ratio: 0.8
   ```

### Implementation Details

The WLASL dataset processing is implemented in two main files:

1. **VideoPreprocessor class** (`src/data/data_preprocessor.py`):
   - `load_wlasl_json()`: Loads and parses the WLASL annotation file
   - `process_wlasl_dataset()`: Processes the entire WLASL dataset
   - `process_wlasl_video()`: Processes a single WLASL video
   - `save_missing_videos_list()`: Saves information about missing videos

2. **Main preprocessing module** (`src/data_preprocessing.py`):
   - `main()`: Main entry point for dataset processing
   - `find_missing_wlasl_videos()`: Specialized function to only find missing videos

### Usage Examples

1. **Process the entire WLASL dataset**:
   ```bash
   python run.py mode.selected_model=data_preprocessor
   ```

2. **Only find missing videos**:
   ```bash
   python run.py mode.selected_model=data_preprocessor mode.find_missing=true
   ```

3. **Process with custom parameters**:
   ```bash
   python run.py mode.selected_model=data_preprocessor data_preprocessing.video.target_fps=25 data_preprocessing.video.max_frames=120
   ```

### Output Files

The processing generates several output files:

1. **Processed video data**: `.npy` files in `data/processed/<gloss>/` directories
2. **Classes list**: `data/processed/classes.txt` containing all processed sign classes
3. **Statistics**: `data/processed/dataset_stats.txt` with dataset statistics
4. **Missing videos log**: `data/missing_videos.txt` listing all missing videos

## Naming Conventions

We follow specific naming conventions to maintain consistency and clarity throughout the codebase:

### General Python Conventions

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use snake_case for variables, functions, and file names
- Use CamelCase for class names
- Use UPPER_CASE for constants

### Project-Specific Conventions

1. **Class Names**:
   - Use descriptive names for classes that perform actions
   - Example: `VideoPreprocessor`, `FeatureExtractor`, `ModelTrainer`, `DataVisualizer`

2. **Module Names**:
   - Class implementation files should be named after the class they contain
   - Example: Class `VideoPreprocessor` would be in module `data_preprocessor.py`
   - Main entry point modules should be named after their function
   - Example: `data_preprocessing.py` for data preprocessing main module

3. **Function Names**:
   - Start with verbs for functions that perform actions
   - Example: `extract_features()`, `preprocess_video()`, `compute_metrics()`

4. **File Naming**:
   - Use snake_case for all file names
   - Implementation files should be named for their primary class: `data_preprocessor.py`, `feature_extractor.py`
   - Main entry point files should be named for their functional area: `data_preprocessing.py`, `feature_extraction.py`
   - Abstract base classes should be prefixed with "base_": `base_model.py`, `base_dataset.py`

5. **Configuration Files**:
   - Configuration files are organized in a hierarchical structure
   - Main configuration: `config.yaml`
   - Mode configurations: `mode/default.yaml`, `mode/test.yaml` 
   - Model configurations: `model/data_preprocessing.yaml`, etc.

## Running the Application

The application uses a single entry point with Hydra configuration:

```bash
# Basic usage
python run.py

# Override the selected model to run
python run.py mode.selected_model=data_preprocessor

# Override specific parameters
python run.py data_preprocessing.video.target_fps=30

# Run tests
python run.py mode=test
```

Model names in the command must correspond to entries in the `model_dispatch` section of the main configuration file.

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
   python run.py mode=test
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
from src.data.data_preprocessor import VideoPreprocessor
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
def test_preprocess_video():
    # Arrange
    video_path = "tests/test_data/sample_video.mp4"
    preprocessor = VideoPreprocessor(target_fps=30, target_height=224, target_width=224)
    
    # Act
    frames = preprocessor.preprocess_video(video_path)
    
    # Assert
    assert frames.shape[0] > 0  # Should have some frames
    assert frames.shape[1] == 224  # Height
    assert frames.shape[2] == 224  # Width
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

If you create new class, you must add `logger` as a parameter:

```python
from src.utils import get_logger
from omegaconf import DictConfig

class ModelTrainer:
    def __init__(self, cfg: DictConfig, logger=None):
        self.logger = logger or get_logger(__name__)
        # Implementation...
```

## Configuration Management

- Store all configurable parameters in YAML files under `configs/`
- Configuration is organized hierarchically:
  - `config.yaml`: Main configuration
  - `mode/`: Contains mode-specific settings
  - `model/`: Contains model-specific settings
  - `wandb/`: Contains WandB settings
- Access configuration using the provided utilities through Hydra
- **Don't hardcode parameters that should be configurable**

## Experiment Tracking

Use Weights & Biases (wandb) for experiment tracking:

```python
import wandb

# Wandb is initialized in run.py if enabled
# Just use wandb.log in your code

# Log metrics during training
wandb.log({"train_loss": loss, "val_accuracy": accuracy})

# Log artifacts
wandb.save("model.pt")
```

## Multi-Platform Considerations

- Use `os.path` for file path operations to ensure cross-platform compatibility
- Avoid platform-specific libraries when possible
- Test code on multiple platforms when feasible
- Use conda environments with appropriate CUDA versions for development

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [WLASL Dataset](https://github.com/dxli94/WLASL) - Word-Level American Sign Language dataset 