import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
class UserInteractionTracker:
    """Track user interactions for retraining"""
    
    def __init__(self, interactions_path="data/user_interactions"):
        self.interactions_path = Path(interactions_path)
        self.interactions_path.mkdir(parents=True, exist_ok=True)
        self.interactions_file = self.interactions_path / 'new_ratings.csv'
        
        # Create file if doesn't exist
        if not self.interactions_file.exists():
            pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp']).to_csv(
                self.interactions_file, index=False
            )
    
    def add_rating(self, user_id, movie_id, rating):
        """Add new user rating"""
        timestamp = int(datetime.now().timestamp())
        
        new_rating = pd.DataFrame([{
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating,
            'timestamp': timestamp
        }])
        
        # Append to file
        new_rating.to_csv(self.interactions_file, mode='a', 
                         header=False, index=False)
        
        logger.info(f"Added rating: User {user_id}, Movie {movie_id}, Rating {rating}")
    
    def get_new_interactions_count(self):
        """Get count of new interactions"""
        df = pd.read_csv(self.interactions_file)
        return len(df)
    
    def merge_with_existing(self, existing_ratings_path):
        """Merge new ratings with existing ones"""
        new_ratings = pd.read_csv(self.interactions_file)
        existing_ratings = pd.read_csv(existing_ratings_path)
        
        # Combine
        combined = pd.concat([existing_ratings, new_ratings], ignore_index=True)
        
        # Remove duplicates (keep latest)
        combined = combined.sort_values('timestamp').drop_duplicates(
            subset=['userId', 'movieId'], keep='last'
        )
        
        return combined
    
    def clear_interactions(self):
        """Clear tracked interactions after retraining"""
        pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp']).to_csv(
            self.interactions_file, index=False
        )
        logger.info("Cleared user interactions")
