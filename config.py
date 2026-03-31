"""Configuration constants for the AI/ML project."""

import os

# Directory for storing logs
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Data paths
PROCESSED_MUSIC_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "music_with_mood.csv")
RAW_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "dataset", "tracks.csv")

# Feature columns for model training
FEATURE_COLUMNS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

# Target column for model training
TARGET_COLUMN = "mood"

# Model paths
MOOD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "mood_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")

# Recommendation parameters
TOP_N_RECOMMENDATIONS = 10