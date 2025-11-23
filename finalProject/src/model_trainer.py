from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import json


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
        """Train collaborative filtering model"""
        logger.info("Training collaborative filtering model...")
        
        # Prepare data
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Train SVD
        svd = SVD(n_factors=100, n_epochs=20, random_state=42)
        
        # Cross-validate
        cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], 
                                    cv=5, verbose=False)
        
        # Train on full dataset
        trainset = data.build_full_trainset()
        svd.fit(trainset)
        
        # Save model
        model_path = self.models_path / 'collaborative_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(svd, f)
        
        metrics = {
            'rmse': float(np.mean(cv_results['test_rmse'])),
            'mae': float(np.mean(cv_results['test_mae'])),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Collaborative model saved. RMSE: {metrics['rmse']:.4f}")
        
        # Save training history
        self.training_history.append(metrics)
        self._save_training_history()
        
        return svd, metrics
    
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
