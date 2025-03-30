# Sign Language Recognition System

## Project Overview

This project aims to develop a deep learning-based video sign language recognition system capable of identifying sign language gestures from video sequences. The system is implemented using PyTorch and supports multi-platform development environments.

## Features

- Video preprocessing and feature extraction
- Transformer-based sequence modeling
- Skeleton point extraction and integration
- Multi-platform compatibility (Windows/Linux/macOS)
- Comprehensive evaluation and visualization tools
- WLASL dataset support with missing files handling

## Environment Requirements

- Python 3.10+
- PyTorch 2.6.0+
- CUDA 12.4+ (for GPU training)
- OpenCV 4.5+ (for video processing)
- See `requirements.txt` for complete dependencies

## Quick Start

### Development Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/cyang2001/ISEP_DL_projet2025.git
cd sign_language_recognition

# Create conda environment
conda create -n sign_language python=3.10
conda activate sign_language

# Install dependencies
pip install -r requirements.txt

```

### Using Docker (For Inference/Deployment)

```bash
# Clone github repository first
git clone https://github.com/cyang2001/ISEP_DL_projet2025.git

cd sign_language_recognition
# Build Docker image
docker login # First time of using docker
docker build -t sign_language_recognition .

# Run container
docker run -it --gpus all -v $(pwd):/workspace sign_language_recognition
```

### VSCode Development Environment(Using Docker)

If you use VSCode, you need to install `Dev Containers` and follow the usage instructions for `Dev Containers`. I have already created a folder `.devcontainer` at the root; you can use it if you want. **However, it won't install `requirements.txt` in the Dev Container.** 

You have to choose **"Create virtual environment"** when you select the Python interpreter. I recommend choosing `conda` and selecting `Python 3.10`. Finally, you only need to install `requirements.txt` manually. Just use:
```bash
pip install -r requirements.txt
pip install -e .
```

## Working with WLASL Dataset

This project supports the [WLASL dataset](https://github.com/dxli94/WLASL) for sign language recognition.

### WLASL Setup

1. Download the WLASL dataset:
   ```bash
   git clone https://github.com/dxli94/WLASL.git
   cd WLASL/start_kit
   python video_downloader.py
   python preprocess.py
   ```

2. Place the downloaded video files in the `data/raw/WLASL` directory
3. Place the annotation file (e.g. `WLASL_v0.3.json`) in the `data` directory
4. Update the configuration in `configs/config.yaml` if needed

### Processing WLASL Dataset

```bash
# Process WLASL dataset with default settings
python run.py mode.selected_model=data_preprocessor

# Check for missing videos without processing
python run.py mode.selected_model=data_preprocessor mode.find_missing=true

# Process with specific parameters
python run.py mode.selected_model=data_preprocessor data_preprocessing.video.target_fps=25
```

For detailed instructions on working with WLASL, see the [WLASL Processing Guide](docs/wlasl_guide.md).

### Handling Missing Videos

Over time, some videos in the WLASL dataset may become unavailable. The system automatically:

1. Detects missing videos during processing
2. Logs them to `data/missing_videos.txt`
3. Generates dataset statistics in `data/processed/dataset_stats.txt`

To request missing videos from the original authors, follow the instructions at:
https://github.com/dxli94/WLASL#requesting-missing--pre-processed-videos

## Usage

All operations use the central `run.py` entry point with Hydra configuration:

### Data Preprocessing

```bash
# Using default configuration
python run.py

# Selecting data preprocessing module
python run.py mode.selected_model=data_preprocessor

# Overriding specific parameters
python run.py data_preprocessing.video.target_fps=30
```

### Training Models

```bash
python run.py mode.selected_model=transformer_trainer
```

### Evaluating Models

```bash
python run.py mode.selected_model=model_evaluator model.transformer.checkpoint_path=saved_models/model.pt
```

### Running Tests

```bash
python run.py mode=test
```

## Project Structure

```
sign_language_recognition/
├── run.py                  # Main entry point
├── src/                    # Source code
│   ├── utils/              # Utility functions
│   ├── data/               # Data processing modules
│   │   ├── data_preprocessor.py
│   │   └── ...
│   ├── data_preprocessing.py     # Main module for preprocessing
│   ├── models/             # Model definitions
│   └── ...
├── configs/                # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── mode/               # Runtime modes
│   ├── model/              # Model configurations
│   └── wandb/              # WandB settings
├── docs/                   # Documentation
│   └── wlasl_guide.md      # WLASL dataset guide
├── tests/                  # Unit tests
└── data/                   # Data directory (gitignored)
    ├── raw/                # Raw video files
    │   └── WLASL/          # WLASL dataset videos
    ├── processed/          # Processed data files
    └── WLASL_v0.3.json     # WLASL dataset annotations
```

See [Architecture Documentation](docs/architecture.md) for more details and [Development Guide](docs/api/development_guide.md) for contributor guidelines.

## Documentation

- [Development Guide](docs/api/development_guide.md) - Guidelines for contributors
- [WLASL Processing Guide](docs/wlasl_guide.md) - Detailed guide for working with WLASL dataset
- [Architecture Documentation](docs/architecture.md) - System architecture details

## License

[MIT License](LICENSE)

## Credits

- [WLASL Dataset](https://github.com/dxli94/WLASL) - Original dataset and preprocessing code by Dongxu Li et al.

## Contact

For questions, please contact the project supervisor. 