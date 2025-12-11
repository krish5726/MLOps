

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRecommender:
    """Main recommender system"""
    
    def __init__(self, models_path="models/saved_models"):
        self.models_path = Path(models_path)
        self.content_model = None
        self.collab_model = None
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        logger.info("Loading models...")
        
        # Load content-based model
        content_path = self.models_path / 'content_based_model.pkl'
        if content_path.exists():
            with open(content_path, 'rb') as f:
                self.content_model = pickle.load(f)
            logger.info("Content-based model loaded")
        
        # Load collaborative model
        collab_path = self.models_path / 'collaborative_model.pkl'
        if collab_path.exists():
            with open(collab_path, 'rb') as f:
                self.collab_model = pickle.load(f)
            logger.info("Collaborative model loaded")
    
    def get_content_recommendations(self, title, n=10):
        """Get content-based recommendations"""
        if self.content_model is None:
            return "Model not loaded!"
        
        try:
            idx = self.content_model['indices'][title]
            sim_scores = list(enumerate(
                self.content_model['similarity_matrix'][idx]
            ))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n+1]
            
            movie_indices = [i[0] for i in sim_scores]
            recommendations = self.content_model['movies_df'].iloc[movie_indices]
            
            return recommendations[['title', 'genres', 'overview', 'vote_average']]
        
        except KeyError:
            return f"Movie '{title}' not found in database!"
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for user-movie pair"""
        if self.collab_model is None:
            return "Model not loaded!"
        
        try:
            # Get indices
            user_index = self.collab_model['user_index']
            movie_index = self.collab_model['movie_index']
            
            if user_id not in user_index or movie_id not in movie_index:
                # Return average rating if user or movie not in training data
                return 3.0
            
            user_idx = user_index.index(user_id)
            movie_idx = movie_index.index(movie_id)
            
            # Predict using NMF
            W = self.collab_model['W']
            H = self.collab_model['H']
            
            prediction = np.dot(W[user_idx], H[:, movie_idx])
            prediction = np.clip(prediction, 0.5, 5.0)
            
            return round(float(prediction), 2)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 3.0  # Return average rating on error
    
    def get_top_n_recommendations(self, user_id, n=10, ratings_df=None):
        """Get top N recommendations for a user"""
        if self.collab_model is None or ratings_df is None:
            return "Model or ratings data not available!"
        
        try:
            user_index = self.collab_model['user_index']
            movie_index = self.collab_model['movie_index']
            
            if user_id not in user_index:
                return f"User {user_id} not found in training data!"
            
            user_idx = user_index.index(user_id)
            
            # Get all predictions for this user
            W = self.collab_model['W']
            H = self.collab_model['H']
            user_predictions = np.dot(W[user_idx], H)
            
            # Get movies already rated by user
            rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
            
            # Create list of (movie_id, predicted_rating) for unrated movies
            predictions = []
            for i, movie_id in enumerate(movie_index):
                if movie_id not in rated_movies:
                    pred = np.clip(user_predictions[i], 0.5, 5.0)
                    predictions.append((movie_id, float(pred)))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions[:n]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return f"Error: {str(e)}"


if __name__ == "__main__":
    print("Alternative Recommender loaded successfully!")
    print("This version works with NMF-based collaborative filtering")
