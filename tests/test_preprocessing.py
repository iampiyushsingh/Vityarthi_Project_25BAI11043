"""
tests/test_preprocessing.py
Tests for src/data_preprocessing.py
Run with: pytest tests/
"""
import os
import sys
import pandas as pd
import pytest

# Ensure local project config is used when tests are run directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import PROCESSED_MUSIC_PATH, FEATURE_COLUMNS, TARGET_COLUMN
 
 
# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Creates a small fake music dataframe for testing."""
    data = {
        "danceability": [0.8, 0.3, 0.6],
        "energy":       [0.9, 0.4, 0.7],
        "loudness":     [-3.0, -8.0, -5.0],
        "speechiness":  [0.05, 0.1, 0.04],
        "acousticness": [0.1, 0.9, 0.3],
        "instrumentalness": [0.0, 0.5, 0.1],
        "liveness":     [0.1, 0.2, 0.15],
        "valence":      [0.8, 0.2, 0.5],
        "tempo":        [120.0, 80.0, 100.0],
        "mood":         ["happy", "sad", "neutral"]
    }
    return pd.DataFrame(data)
 
 
# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────
def test_feature_columns_exist(sample_df):
    """All expected feature columns should be present in the dataframe."""
    for col in FEATURE_COLUMNS:
        assert col in sample_df.columns, f"Missing column: {col}"
 
 
def test_target_column_exists(sample_df):
    """Target column (mood) should exist."""
    assert TARGET_COLUMN in sample_df.columns
 
 
def test_no_null_values(sample_df):
    """Dataset should have no nulls after preprocessing."""
    assert sample_df.isnull().sum().sum() == 0
 
 
def test_processed_file_exists():
    """Processed CSV should exist after preprocessing pipeline runs."""
    assert os.path.isfile(PROCESSED_MUSIC_PATH), (
        f"Processed file not found: {PROCESSED_MUSIC_PATH}. "
        "Run data_preprocessing.py first.")