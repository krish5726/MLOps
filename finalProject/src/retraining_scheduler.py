

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train both content-based and collaborative filtering models"""
    
    def __init__(self, models_path="models/saved_models"):
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.training_history = []
        
    def train_content_based(self, movies_df):
        """Train content-based model"""
        logger.info("Training content-based model...")
        
        # Create count vectorizer
        count = CountVectorizer(stop_words='english', max_features=10000)
        count_matrix = count.fit_transform(movies_df['soup_str'])
        
        # Calculate similarity
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
        
        # Create indices
        movies_df = movies_df.reset_index(drop=True)
        indices = pd.Series(movies_df.index, index=movies_df['title'])
        
        # Save model artifacts
        model_data = {
            'similarity_matrix': cosine_sim,
            'indices': indices,
            'movies_df': movies_df[['id', 'title', 'genres', 'overview', 
                                   'vote_average', 'popularity']],
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.models_path / 'content_based_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Content-based model saved to {model_path}")
        return model_data
    
    def train_collaborative(self, ratings_df):
        """Train collaborative filtering model using NMF"""
        logger.info("Training collaborative filtering model using NMF...")
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        
        # Split data for evaluation
        train_mask = np.random.rand(len(ratings_df)) < 0.8
        train_data = ratings_df[train_mask]
        test_data = ratings_df[~train_mask]
        
        # Create training matrix
        train_matrix = train_data.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Ensure same shape
        train_matrix = train_matrix.reindex(
            index=user_item_matrix.index,
            columns=user_item_matrix.columns,
            fill_value=0
        )
        
        # Train NMF model
        n_components = min(50, min(train_matrix.shape) - 1)
        nmf_model = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            max_iter=200
        )
        
        W = nmf_model.fit_transform(train_matrix)
        H = nmf_model.components_
        
        # Predictions
        predicted_ratings = np.dot(W, H)
        
        # Evaluate on test set
        test_predictions = []
        test_actuals = []
        
        for _, row in test_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            
            if user_id in user_item_matrix.index and movie_id in user_item_matrix.columns:
                user_idx = user_item_matrix.index.get_loc(user_id)
                movie_idx = user_item_matrix.columns.get_loc(movie_id)
                
                pred = predicted_ratings[user_idx, movie_idx]
                # Clip predictions to valid range
                pred = np.clip(pred, 0.5, 5.0)
                
                test_predictions.append(pred)
                test_actuals.append(row['rating'])
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae = mean_absolute_error(test_actuals, test_predictions)
        
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Save model
        model_data = {
            'nmf_model': nmf_model,
            'W': W,
            'H': H,
            'user_item_matrix': user_item_matrix,
            'user_index': user_item_matrix.index.tolist(),
            'movie_index': user_item_matrix.columns.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        model_path = self.models_path / 'collaborative_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Collaborative model saved to {model_path}")
        
        # Save training history
        self.training_history.append(metrics)
        self._save_training_history()
        
        return model_data, metrics
    
    def _save_training_history(self):
        """Save training history"""
        history_path = Path("logs/training_logs")
        history_path.mkdir(parents=True, exist_ok=True)
        
        with open(history_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def train_all(self, movies_df, ratings_df):
        """Train all models"""
        content_model = self.train_content_based(movies_df)
        collab_model, metrics = self.train_collaborative(ratings_df)
        
        return {
            'content_based': content_model,
            'collaborative': collab_model,
            'metrics': metrics
        }


if __name__ == "__main__":
    print("Alternative Model Trainer loaded successfully!")
    print("This version uses NMF instead of SVD from scikit-surprise")
