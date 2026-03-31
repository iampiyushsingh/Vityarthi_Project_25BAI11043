import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import load_and_clean_data, create_mood_labels, save_processed_data

def prepare_data(df):
    """Prepare data for training."""
    features = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    X = df[features]
    y = df['mood']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("✅ Data Split Done. Training:", X_train.shape[0], "Testing:", X_test.shape[0])
    return X_train, X_test, y_train, y_test, le, features

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model Trained!")
    return model

def evaluate_model(model, X_test, y_test, le):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return accuracy

def save_model(model, le, model_path='models/mood_model.pkl', encoder_path='models/label_encoder.pkl'):
    """Save the trained model and label encoder."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Label encoder saved to {encoder_path}")

def train_pipeline(data_path='data/raw/dataset/tracks.csv'):
    """Complete training pipeline."""
    # Load and preprocess data
    df = load_and_clean_data(data_path)
    df = create_mood_labels(df)
    save_processed_data(df)

    # Prepare and train
    X_train, X_test, y_train, y_test, le, features = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, le)
    save_model(model, le)

    return model, le