import pandas as pd
import os

def load_and_clean_data(data_path='data/raw/dataset/tracks.csv'):
    """Load and clean the music dataset."""
    df = pd.read_csv(data_path)

    # Drop unnecessary columns if they exist
    columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Remove duplicates and missing values
    df = df.drop_duplicates()
    df = df.dropna()

    print(f"✅ Data Loaded and Cleaned. Shape: {df.shape}")
    return df

def assign_mood(row):
    """Assign mood based on multiple audio features for more realistic classification."""
    valence = row['valence']      # 0-1: negative to positive
    energy = row['energy']        # 0-1: calm to energetic
    danceability = row['danceability']  # 0-1: not danceable to very danceable
    acousticness = row['acousticness']  # 0-1: electronic to acoustic
    loudness = row['loudness']    # -60 to 0: quiet to loud
    tempo = row['tempo']          # BPM

    # Normalize loudness to 0-1 scale
    loudness_norm = (loudness + 60) / 60

    # Calculate mood scores based on multiple features
    happy_score = (valence * 0.4 + energy * 0.3 + danceability * 0.2 + loudness_norm * 0.1)
    sad_score = ((1 - valence) * 0.4 + (1 - energy) * 0.3 + acousticness * 0.2 + (1 - loudness_norm) * 0.1)
    angry_score = ((1 - valence) * 0.4 + energy * 0.3 + (1 - acousticness) * 0.2 + loudness_norm * 0.1)
    calm_score = (valence * 0.3 + (1 - energy) * 0.3 + acousticness * 0.2 + (1 - tempo/200) * 0.2)

    # Return mood with highest score
    scores = {
        'Happy': happy_score,
        'Sad': sad_score,
        'Angry': angry_score,
        'Calm': calm_score
    }

    return max(scores, key=scores.get)

def create_mood_labels(df):
    """Add mood column to dataframe."""
    df['mood'] = df.apply(assign_mood, axis=1)
    print("✅ Moods Created:")
    print(df['mood'].value_counts())
    return df

def save_processed_data(df, output_path='data/processed/music_with_mood.csv'):
    """Save the processed dataframe with moods."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")