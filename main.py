#!/usr/bin/env python3
"""
Music Mood Recommendation System
Main entry point for training and recommendation
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.model_training import train_pipeline
from src.recommendation import run_recommendation

def main():
    parser = argparse.ArgumentParser(
        description='Music Mood Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train the model
  python main.py recommend                # Get recommendations (interactive)
  python main.py recommend --mood Happy   # Get Happy songs
        """
    )
    parser.add_argument('action', nargs='?', default='recommend', choices=['train', 'recommend'],
                       help='Action to perform: train the model or get recommendations (default: recommend)')
    parser.add_argument('--mood', choices=['Happy', 'Sad', 'Angry', 'Calm'],
                       help='Mood for recommendations (optional, will prompt if not provided)')

    args = parser.parse_args()

    if args.action == 'train':
        print("Starting model training...")
        train_pipeline()
        print("Training completed!")
    elif args.action == 'recommend':
        run_recommendation(args.mood)

if __name__ == '__main__':
    main()