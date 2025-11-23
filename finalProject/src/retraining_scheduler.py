class RetrainingScheduler:
    """Manage model retraining"""
    
    def __init__(self, threshold=100):
        self.threshold = threshold  # Retrain after N new interactions
        self.tracker = UserInteractionTracker()
        self.trainer = ModelTrainer()
        
    def check_and_retrain(self, movies_df):
        """Check if retraining is needed and execute"""
        new_count = self.tracker.get_new_interactions_count()
        
        logger.info(f"New interactions: {new_count}")
        
        if new_count >= self.threshold:
            logger.info(f"Threshold reached ({new_count} >= {self.threshold}). Starting retraining...")
            
            # Merge ratings
            combined_ratings = self.tracker.merge_with_existing(
                'data/processed/ratings_processed.csv'
            )
            
            # Retrain models
            self.trainer.train_all(movies_df, combined_ratings)
            
            # Update processed ratings
            combined_ratings.to_csv(
                'data/processed/ratings_processed.csv', index=False
            )
            
            # Clear interactions
            self.tracker.clear_interactions()
            
            logger.info("Retraining completed!")
            return True
        
        return False


