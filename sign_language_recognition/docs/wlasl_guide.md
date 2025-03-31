# WLASL Dataset Processing Guide

This guide provides detailed instructions on how to use our Sign Language Recognition system with the WLASL (Word-Level American Sign Language) dataset.

## Introduction to WLASL

[WLASL](https://github.com/dxli94/WLASL) (Word-Level American Sign Language) is a large-scale dataset for word-level sign language recognition, developed by Dongxu Li et al. The dataset contains:

- Over 2000 sign glosses (words)
- More than 21,000 video samples
- Videos sourced from various online platforms
- Annotations in JSON format with frame-level details

## Setup Instructions

### 1. Download the WLASL Dataset

First, clone the WLASL repository and download the raw videos:

```bash
# Clone the WLASL repository
git clone https://github.com/dxli94/WLASL.git

# Navigate to the start_kit directory
cd WLASL/start_kit

# Download videos (this may take several hours)
python video_downloader.py

```

### 2. Prepare the Dataset for Our System

After downloading the WLASL dataset, you need to:

1. Create the necessary directory structure:
   ```bash
   mkdir -p sign_language_recognition/data/raw/WLASL
   mkdir -p sign_language_recognition/data/processed
   ```

2. Copy the video files to our project structure:
   ```bash
   # Copy the downloaded videos to our raw directory
   cp -r WLASL/videos/* sign_language_recognition/data/raw/WLASL/
   
   # Copy the annotation file
   cp WLASL/start_kit/WLASL_v0.3.json sign_language_recognition/data/
   ```

3. Verify the configuration in `sign_language_recognition/configs/config.yaml`:
   ```yaml
   data:
     dataset: "wlasl"
     raw_dir: "data/raw/WLASL"
     processed_dir: "data/processed"
     wlasl_json_path: "data/WLASL_v0.3.json"
     missing_videos_log: "data/missing_videos.txt"
   ```

## Processing the Dataset

### Check for Missing Videos

Before processing the entire dataset, it's a good idea to check for missing videos:

```bash
python run.py mode.selected_model=data_preprocessor mode.find_missing=true
```

This will:
- Scan the WLASL JSON file
- Check if each video exists in your `data/raw/WLASL` directory
- Generate a list of missing videos in `data/missing_videos.txt`

### Process the Dataset

To process the entire dataset with default settings:

```bash
python run.py mode.selected_model=data_preprocessor
```

This will:
1. Read the WLASL JSON annotations
2. Extract frames from each video
3. Apply preprocessing (resizing, normalization, etc.)
4. Save processed videos as numpy arrays in the `data/processed` directory
5. Generate dataset statistics and class lists

### Customizing Preprocessing Parameters

You can customize various preprocessing parameters:

```bash
# Change video dimensions and maximum frames
python run.py mode.selected_model=data_preprocessor \
    data_preprocessing.video.target_height=224 \
    data_preprocessing.video.target_width=224 \
    data_preprocessing.video.max_frames=150

# Change cropping method
python run.py mode.selected_model=data_preprocessor \
    data_preprocessing.video.crop_method=center  # Options: center, random, none
```

## Handling Missing Videos

Over time, some videos in the WLASL dataset may become unavailable as URLs expire. Our system handles this by:

1. Automatically detecting missing videos
2. Logging them to `data/missing_videos.txt`
3. Continuing processing with available videos

To request missing videos from the original authors:

1. Generate the list of missing videos as described above
2. Follow the instructions at: https://github.com/dxli94/WLASL#requesting-missing--pre-processed-videos
3. Submit a request form to the authors
4. Once you receive the missing videos, place them in your `data/raw/WLASL` directory
5. Run the processing again

## Output Structure

After processing, your data structure will look like:

```
data/
├── raw/
│   └── WLASL/
│       ├── <video_id>.mp4
│       └── ...
├── processed/
│   ├── <gloss1>/
│   │   ├── <gloss1>_1.npy
│   │   ├── <gloss1>_2.npy
│   │   └── ...
│   ├── <gloss2>/
│   │   └── ...
│   ├── classes.txt        # List of all glosses
│   └── dataset_stats.txt  # Dataset statistics
├── WLASL_v0.3.json        # Original annotations
└── missing_videos.txt     # List of missing videos
```

Each `.npy` file contains the preprocessed frames for a single video sample.

## Using the Processed Data

The processed data is ready to be used for training sign language recognition models:

```bash
# Train a transformer model on the processed data
python run.py mode.selected_model=transformer_trainer
```

## References

- [WLASL Dataset GitHub Repository](https://github.com/dxli94/WLASL)
- [WLASL Paper: Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://openaccess.thecvf.com/content_WACV_2020/papers/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.pdf)

## Citation

If you use the WLASL dataset in your research, please cite the original paper:

```
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
``` 
