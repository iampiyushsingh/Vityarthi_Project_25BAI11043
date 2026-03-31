"""
Quick script to train and save mood classification models for testing.
Run with: python scripts/train_models.py
"""
import os
import sys
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
 
# Ensure project root is on path so local config.py is imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
 
from config import (
    PROCESSED_MUSIC_PATH, FEATURE_COLUMNS, TARGET_COLUMN,
    MOOD_MODEL_PATH, LABEL_ENCODER_PATH
)
 
 
def train_and_save():
    # Ensure models directory exists
    os.makedirs(os.path.dirname(MOOD_MODEL_PATH), exist_ok=True)
 
    # Load processed data
    df = pd.read_csv(PROCESSED_MUSIC_PATH)
 
    # Extract features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
 
    # Handle missing values
    X = X.fillna(X.mean())
 
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
 
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
 
    # Train model (n_estimators=50 for speed during testing)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
 
    # Save model
    with open(MOOD_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {MOOD_MODEL_PATH}")
 
    # Save label encoder
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Label encoder saved to: {LABEL_ENCODER_PATH}")
 
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_score:.3f}")
    print(f"Test Accuracy:  {test_score:.3f}")
 
 
if __name__ == "__main__":   # ← prevents auto-run if accidentally imported
    train_and_save()