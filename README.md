# AIML Vityarthi BYOP — 25BAI11043
#  Music Mood Recommendation System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)


## Overview

This project builds a **mood-based music recommendation engine** using supervised machine learning. It takes raw Spotify track data, assigns mood labels based on audio features, trains a Random Forest classifier, and recommends songs that match a chosen mood.

-  Mood labels: **Happy**, **Sad**, **Angry**, **Calm**
-  Classifier: **Random Forest** (9 audio features)
-  Training data: **113,549 songs**
-  Test Accuracy: **99.1%**

## Key Features

- Dataset ingestion, cleaning, and mood-label generation
- Random Forest classifier trained on Spotify audio features
- Mood-based song recommendations via CLI or Python API
- Centralized configuration via `config.py`
- Shared utilities for logging and model I/O via `utils.py`
- Full test coverage with `pytest`

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Files](#model-files)
- [Workflow](#workflow)
- [Model Performance](#model-performance)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)
- [Notes](#notes)
- [Key Functions](#key-functions)

## Project Structure

```
aiml-project/
├── data/
│   ├── raw/dataset/
│   │   └── tracks.csv                   # Source raw track data
│   └── processed/
│       └── music_with_mood.csv          # Preprocessed with mood labels
├── models/
│   ├── mood_model.pkl                   # Serialized classifier
│   └── label_encoder.pkl               # Mood label encoder
├── scripts/
│   └── train_models.py                  # Lightweight training script
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Load/process pipeline
│   ├── model_training.py               # Train, evaluate, save pipeline
│   └── recommendation.py               # Recommend tracks by mood
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_recommendation.py
├── .gitignore
├── README.md
├── config.py                           # Adjustable settings
├── main.py  
├── Project_report.pdf                           # CLI entrypoint
├── pytest.ini
├── requirements.txt
└── utils.py                            # Helper functions
```

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/iampiyushsingh/Vityarthi_Project_25BAI11043.git
   cd "Vityarthi_Project_25BAI11043"
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.7+ |
| pandas | ≥1.3.0 |
| numpy | ≥1.21.0 |
| scikit-learn | ≥1.0.0 |
| matplotlib | ≥3.5.0 |
| seaborn | ≥0.11.0 |
| pytest | ≥7.0.0 |

- **Hardware**: Standard PC with at least 4 GB RAM
- **Storage**: ~500 MB for dataset and model files

## Data Preparation

- Input dataset path: `data/raw/dataset/tracks.csv`
- Output processed dataset: `data/processed/music_with_mood.csv`

`src/data_preprocessing.py` handles feature selection, mood-label creation, and saving cleaned data.

## Training

Run the training via CLI:

```bash
python main.py train
```

Or use the lightweight script:

```bash
python scripts/train_models.py
```

This pipeline:
- Loads processed data
- Encodes mood labels
- Trains a Random Forest classifier
- Evaluates accuracy, precision, recall, F1
- Saves `models/mood_model.pkl` and `models/label_encoder.pkl`

## How It Works

### Data Preprocessing
- Loads raw track data from CSV
- Cleans missing values and duplicates
- Derives mood labels from audio features
- Saves processed data for training

### Model Training
- Uses Random Forest classifier
- Features: valence, energy, danceability, etc.
- Encodes mood labels (Happy, Sad, Angry, Calm)
- Trains on 80% data, validates on 20%

### Recommendation
- Predicts mood for new tracks
- Filters and ranks recommendations by predicted mood

### Audio Features Used

| Feature | Description |
|---------|-------------|
| `valence` | Musical positivity |
| `energy` | Intensity and activity |
| `danceability` | Suitability for dancing |
| `acousticness` | Amount of acoustic sound |
| `loudness` | Overall loudness in dB |
| `tempo` | Beats per minute |
| `instrumentalness` | Vocal vs instrumental ratio |
| `speechiness` | Presence of spoken words |
| `liveness` | Presence of live audience |

Mood assignment is computed from aggregated sentiment score heuristics and refined with a RandomForest classifier. This gives robust mood predictions using real track features.

## Model Files

| File | Purpose |
|------|---------|
| `models/mood_model.pkl` | Trained RandomForest model for mood prediction |
| `models/label_encoder.pkl` | Encodes mood labels (Happy, Sad, Angry, Calm) to numeric values |

## Workflow

```
Raw Data (tracks.csv) → Data Preprocessing → Processed Data (music_with_mood.csv)
                                                            ↓
Model Training (main.py train) → Saved Models (.pkl files)
                                                            ↓
Recommendation (main.py recommend) → Mood-based Song Recommendations
```

## Usage

### Step-by-Step Guide

1. **Prepare Data**: Ensure `data/raw/dataset/tracks.csv` exists with required columns.
2. **Train Model**: Run `python main.py train` to train and save the model.
3. **Get Recommendations**: Use `python main.py recommend --mood Happy` for CLI or programmatic API.

### CLI: Recommendation

```bash
# Interactive mode
python main.py recommend

# With mood specified
python main.py recommend --mood Happy

# With mood and number of songs
python main.py recommend --mood Happy --top-k 10
```

### Programmatic

```python
from src.recommendation import run_recommendation

run_recommendation(mood="Happy", top_k=10)
```

## Model Performance

- **Train Accuracy**: 1.000
- **Test Accuracy**: 0.991
- **Precision/Recall**: >98% for all mood categories
- **Training Data**: 113,549 songs
- **Features**: 9 audio features

>  A train accuracy of 100% may indicate overfitting. Consider tuning `max_depth` or `min_samples_split` in `config.py` to improve generalization.

## Model and Mood Logic

Mood classification uses the following features (from Spotify API style input):
- valence
- energy
- danceability
- acousticness
- loudness
- tempo
- instrumentalness
- speechiness
- liveness

Mood assignment is computed from aggregated sentiment score heuristics and refined with a RandomForest classifier. This gives robust mood predictions using real track features.

## Evaluation

- Run all tests:
  ```bash
  pytest tests/ -v
  ```
- Run single test module:
  ```bash
  pytest tests/test_model.py -v
  ```
- Run with coverage:
  ```bash
  pytest tests/ --cov=src --cov-report=term-missing
  ```

Metrics tracked:
- accuracy
- precision
- recall
- F1 score

## Contributing

1. Fork repository
2. Create branch `feature/<name>`
3. Add tests and update docs
4. Submit pull request

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: tracks.csv` | Ensure `data/raw/dataset/tracks.csv` exists with required columns |
| `Model not found` error | Run `python main.py train` first |
| `pytest` import errors | Activate venv and run `pip install -r requirements.txt` |
| pytest verbose debug | Run `pytest tests/ -v --maxfail=1` |
| Low recommendation quality | Retrain: `python main.py train` |

## Notes

- The model requires training before recommendations
- Mood categories are fixed: Happy, Sad, Angry, Calm
- All model files are saved for quick predictions without retraining
- Model files are excluded from Git due to size — regenerate with `python main.py train`

## Key Functions

- `preprocess()`: Loads and cleans raw track data, assigns mood labels
- `train_pipeline()`: Trains and saves the Random Forest model
- `run_recommendation()`: Returns mood-based song recommendations
- `get_logger()`: Sets up console and file logging
- `save_model()` / `load_model()`: Handles model persistence

## License

This project is part of an AIML course assignment at VIT Bhopal University.

## Contact

- **Project maintainer**: Piyush Kumar Singh
- **Email**: piyush.25bai11043@vitbhopal.ac.in
- **GitHub**: [@iampiyushsingh](https://github.com/iampiyushsingh)