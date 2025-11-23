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
        
        prediction = self.collab_model.predict(user_id, movie_id)
        return round(prediction.est, 2)
    
    def get_top_n_recommendations(self, user_id, n=10, ratings_df=None):
        """Get top N recommendations for a user"""
        if self.collab_model is None or ratings_df is None:
            return "Model or ratings data not available!"
        
        # Get all movies
        all_movies = ratings_df['movieId'].unique()
        
        # Get movies already rated by user
        rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
        
        # Get unrated movies
        unrated_movies = [m for m in all_movies if m not in rated_movies]
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies[:100]:  # Limit for performance
            pred = self.collab_model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]
