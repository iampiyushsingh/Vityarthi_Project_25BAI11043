import pickle
import pandas as pd
import os

def load_model_and_data(model_path='models/mood_model.pkl',
                       encoder_path='models/label_encoder.pkl',
                       data_path='data/processed/music_with_mood.csv'):
    """Load the trained model, encoder, and processed data."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    df = pd.read_csv(data_path)
    return model, le, df

def recommend_songs(user_mood, df, num_recommendations=10):
    """Recommend songs based on user's mood."""
    # Filter songs matching the mood
    recommended = df[df['mood'] == user_mood][['track_name', 'artists', 'track_genre']]

    if recommended.empty:
        print(f"No songs found for mood: {user_mood}")
        return pd.DataFrame()

    # Return random sample of recommendations
    return recommended.sample(min(num_recommendations, len(recommended)))

def display_recommendations(recommendations, user_mood):
    """Display the recommended songs."""
    print(f"\n🎵 Top {len(recommendations)} songs for your mood ({user_mood}):\n")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))
    else:
        print("No recommendations available.")

def get_user_mood(args_mood=None):
    """Get mood input from user or args."""
    available_moods = ['Happy', 'Sad', 'Angry', 'Calm']

    if args_mood:
        mood = args_mood.strip().capitalize()
        if mood in available_moods:
            return mood
        else:
            raise ValueError(f"Invalid mood: {args_mood}. Must be one of {available_moods}")

    print("🎵 Mood Based Music Recommendation System")
    print("------------------------------------------")
    print(f"Available moods: {', '.join(available_moods)}")

    while True:
        user_mood = input("Enter your mood: ").strip().capitalize()
        if user_mood in available_moods:
            return user_mood
        else:
            print(f"Invalid mood. Please choose from: {', '.join(available_moods)}")

def run_recommendation(mood=None):
    """Run the recommendation system."""
    try:
        model, le, df = load_model_and_data()
        user_mood = get_user_mood(mood)
        recommendations = recommend_songs(user_mood, df)
        display_recommendations(recommendations, user_mood)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training first to create the required files.")
    except ValueError as e:
        print(f"Error: {e}")