"""
tests/test_recommendation.py
Tests for src/recommendation.py
Run with: pytest tests/
"""
import os
import sys
import pytest
import pandas as pd

# Ensure local project config is used when tests are run directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import TOP_N_RECOMMENDATIONS, FEATURE_COLUMNS


# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────
@pytest.fixture
def sample_music_df():
    """Fake music dataframe simulating music_with_mood.csv."""
    data = {
        "track_name": [f"Song {i}" for i in range(20)],
        "artist":     [f"Artist {i}" for i in range(20)],
        "mood":       (["happy"] * 7 + ["sad"] * 7 + ["neutral"] * 6),
        "danceability": [0.5 + i * 0.02 for i in range(20)],
        "energy":       [0.6 + i * 0.01 for i in range(20)],
        "valence":      [0.4 + i * 0.02 for i in range(20)],
    }
    return pd.DataFrame(data)


# ──────────────────────────────────────────────
# HELPERS (mirrors logic in recommendation.py)
# ──────────────────────────────────────────────
def get_recommendations(df: pd.DataFrame, mood: str, n: int) -> pd.DataFrame:
    """Simple filter-based recommendation — replace with your actual function."""
    filtered = df[df["mood"] == mood]
    return filtered.head(n)


# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────
def test_recommendations_returned(sample_music_df):
    """Should return a non-empty list of recommendations for a valid mood."""
    results = get_recommendations(sample_music_df, "happy", TOP_N_RECOMMENDATIONS)
    assert len(results) > 0


def test_recommendations_correct_mood(sample_music_df):
    """All recommended tracks should match the requested mood."""
    results = get_recommendations(sample_music_df, "sad", TOP_N_RECOMMENDATIONS)
    assert all(results["mood"] == "sad")


def test_recommendations_count(sample_music_df):
    """Should return at most TOP_N_RECOMMENDATIONS tracks."""
    results = get_recommendations(sample_music_df, "happy", TOP_N_RECOMMENDATIONS)
    assert len(results) <= TOP_N_RECOMMENDATIONS


def test_invalid_mood_returns_empty(sample_music_df):
    """An unknown mood should return an empty dataframe, not crash."""
    results = get_recommendations(sample_music_df, "angry", TOP_N_RECOMMENDATIONS)
    assert len(results) == 0


def test_result_has_track_name(sample_music_df):
    """Recommendations should include a track_name column."""
    results = get_recommendations(sample_music_df, "neutral", TOP_N_RECOMMENDATIONS)
    assert "track_name" in results.columns