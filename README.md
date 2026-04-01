# AIML Vityarthi BYOP — 25BAI11043
# Music Mood Recommendation System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Academic-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)


## Overview

I built this project to recommend songs based on how you're feeling, not what you've previously listened to. It works by pulling audio features from a Spotify dataset, figuring out the mood of each track, and then matching songs to whatever mood you pick.

- Mood labels: **Happy**, **Sad**, **Angry**, **Calm**
- Classifier: **Random Forest** (9 audio features)
- Training data: **113,549 songs**
- Test Accuracy: **99.1%**

## Key Features

- Loads and cleans the raw Spotify data, then automatically assigns mood labels
- Random Forest classifier trained on Spotify audio features
- Mood-based song recommendations via CLI or Python API
- All paths and settings live in one config.py so nothing is hardcoded
- Logging and model save/load are handled in utils.py so I'm not repeating code everywhere
- Every core module has pytest tests written for it

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
├── Project_report.pdf                  # CLI entrypoint
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
2. Set up a virtual environment:
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

`src/data_preprocessing.py` takes care of picking the right columns, creating the mood labels and saving the cleaned output.

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
- Reads the raw CSV
- Drops rows with missing values and removes duplicate tracks
- Uses valence, energy and other features to assign a mood to each track
- Saves the cleaned file so training doesn't redo this every run

### Model Training
- Uses Random Forest classifier
- Features: valence, energy, danceability, etc.
- Converts mood names to numbers so the model can work with them
- 80% goes to training, 20% held back for testing

### Recommendation
- Predicts mood for new tracks
- Filters the dataset to tracks matching your mood and returns the top results

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

I set mood labels using simple threshold rules — high valence + high energy = Happy, low both = Sad, etc. The classifier picks up on these patterns once trained.

## Model Files

| File | Purpose |
|------|---------|
| `models/mood_model.pkl` | The trained model, saved after running main.py train |
| `models/label_encoder.pkl` | Converts between mood names and the numbers the model uses internally |

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

1. **Prepare Data**: Make sure `data/raw/dataset/tracks.csv` is in place with all the needed columns.
2. **Train Model**: Run `python main.py train` — this trains and saves the model.
3. **Get Recommendations**: Run `python main.py recommend --mood Happy` from CLI or call it directly in Python.

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

> 100% train accuracy probably means it memorised the training data. Test accuracy is still 99.1% so it's not terrible, but I want to try capping `max_depth` or increasing `min_samples_split` in `config.py` to close that gap.

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

I set mood labels using simple threshold rules — high valence + high energy = Happy, low both = Sad, etc. The classifier picks up on these patterns once trained.

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

- Run training first or recommendations won't work
- Moods are fixed to these four: Happy, Sad, Angry, Calm
- After training once the .pkl files stick around, no need to retrain every session
- I put model/*.pkl in .gitignore — they're too big for GitHub, just run `python main.py train` to get them back

## Key Functions

- `preprocess()`: reads the raw CSV, drops bad rows and stamps a mood on each track
- `train_pipeline()`: runs the full training pipeline and saves the model to disk
- `run_recommendation()`: takes a mood input and gives back matching songs
- `get_logger()`: sets up logging so you can see what's happening and check logs later
- `save_model()` / `load_model()`: saves and loads the .pkl files

## License

This project is part of an AIML course assignment at VIT Bhopal University.

## Contact

- **Project maintainer**: Piyush Kumar Singh
- **Email**: piyush.25bai11043@vitbhopal.ac.in
- **GitHub**: [@iampiyushsingh](https://github.com/iampiyushsingh)
