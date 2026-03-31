"""
tests/test_model.py
Tests for src/model_training.py
Run with: pytest tests/
"""
import os
import sys
import pickle
import pytest
import numpy as np

# Ensure local project config is used when tests are run directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import MOOD_MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_COLUMNS
 
 
# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────
@pytest.fixture
def mood_model():
    """Loads the trained mood model from disk."""
    assert os.path.isfile(MOOD_MODEL_PATH), (
        f"Model not found at {MOOD_MODEL_PATH}. Run model_training.py first."
    )
    with open(MOOD_MODEL_PATH, "rb") as f:
        return pickle.load(f)
 
 
@pytest.fixture
def label_encoder():
    """Loads the label encoder from disk."""
    assert os.path.isfile(LABEL_ENCODER_PATH), (
        f"Label encoder not found at {LABEL_ENCODER_PATH}."
    )
    with open(LABEL_ENCODER_PATH, "rb") as f:
        return pickle.load(f)
 
 
@pytest.fixture
def sample_input():
    """A single fake track feature vector for prediction testing."""
    return np.array([[0.8, 0.9, -3.0, 0.05, 0.1, 0.0, 0.1, 0.8, 120.0]])
 
 
# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────
def test_model_file_exists():
    """Trained model .pkl file should exist."""
    assert os.path.isfile(MOOD_MODEL_PATH)
 
 
def test_model_can_predict(mood_model, sample_input):
    """Model should return a prediction without errors."""
    prediction = mood_model.predict(sample_input)
    assert prediction is not None
    assert len(prediction) == 1
 
 
def test_model_predict_proba(mood_model, sample_input):
    """Model should support predict_proba (confidence scores)."""
    proba = mood_model.predict_proba(sample_input)
    assert proba is not None
    assert proba.shape[1] > 1  # more than 1 class
 
 
def test_label_encoder_has_classes(label_encoder):
    """Label encoder should have at least 2 mood classes."""
    assert hasattr(label_encoder, "classes_")
    assert len(label_encoder.classes_) >= 2
 
 
def test_correct_number_of_features(mood_model, sample_input):
    """Model input should match the expected number of features."""
    assert sample_input.shape[1] == len(FEATURE_COLUMNS)