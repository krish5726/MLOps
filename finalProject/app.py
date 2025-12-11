import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data_pipeline import DataPipeline
from model_trainer import ModelTrainer
from recommender import MovieRecommender
from user_interaction_tracker import UserInteractionTracker
from retraining_scheduler import RetrainingScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovieRecommenderApp:
    """Main application class"""
    
    def __init__(self):
        self.recommender = None
        self.tracker = UserInteractionTracker()
        self.scheduler = RetrainingScheduler(threshold=50)  # Retrain after 50 new ratings
        self.movies_df = None
        self.ratings_df = None
        
    def setup(self):
        """Initial setup - run data pipeline and train models"""
        print("\n" + "="*70)
        print("MOVIE RECOMMENDATION SYSTEM - SETUP")
        print("="*70)
        
        # Check if data files exist
        raw_path = Path("data/raw")
        if not (raw_path / "tmdb_5000_movies.csv").exists():
            print("\n❌ ERROR: Data files not found!")
            print("\nPlease place the following files in 'data/raw/' folder:")
            print("  - tmdb_5000_credits.csv")
            print("  - tmdb_5000_movies.csv")
            print("  - ratings_small.csv")
            print("\nDownload from:")
            print("  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
            print("  https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
            return False
        
        # Check if models already exist
        models_path = Path("models/saved_models")
        if (models_path / "content_based_model.pkl").exists():
            print("\n✅ Models already exist. Loading...")
            self.load_data()
            self.recommender = MovieRecommender()
            return True
        
        print("\n🔄 Running initial setup...")
        
        # Run data pipeline
        print("\n1️⃣ Processing data...")
        pipeline = DataPipeline()
        self.movies_df, self.ratings_df = pipeline.run()
        
        # Train models
        print("\n2️⃣ Training models (this may take a few minutes)...")
        trainer = ModelTrainer()
        results = trainer.train_all(self.movies_df, self.ratings_df)
        
        print(f"\n✅ Training completed!")
        print(f"   RMSE: {results['metrics']['rmse']:.4f}")
        print(f"   MAE: {results['metrics']['mae']:.4f}")
        
        # Load recommender
        self.recommender = MovieRecommender()
        
        return True
    
    def load_data(self):
        """Load processed data"""
        processed_path = Path("data/processed")
        self.movies_df = pd.read_pickle(processed_path / 'movies_processed.pkl')
        self.ratings_df = pd.read_csv(processed_path / 'ratings_processed.csv')
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("1. Get movie recommendations (Content-Based)")
        print("2. Predict rating for a movie")
        print("3. Get personalized recommendations (Collaborative)")
        print("4. Add a new rating")
        print("5. View statistics")
        print("6. Check for retraining")
        print("7. Search movies")
        print("8. Exit")
        print("="*70)
    
    def get_recommendations(self):
        """Get content-based recommendations"""
        print("\n" + "-"*70)
        print("CONTENT-BASED RECOMMENDATIONS")
        print("-"*70)
        
        # Show some popular movies
        print("\nPopular movies you can try:")
        popular = self.movies_df.nlargest(10, 'popularity')[['title', 'genres']]
        for idx, row in popular.iterrows():
            genres_str = ', '.join(row['genres'][:3]) if isinstance(row['genres'], list) else 'N/A'
            print(f"  • {row['title']} ({genres_str})")
        
        title = input("\nEnter movie title: ").strip()
        n = int(input("Number of recommendations (default 10): ") or "10")
        
        print(f"\n🎬 Recommendations for '{title}':\n")
        recommendations = self.recommender.get_content_recommendations(title, n)
        
        if isinstance(recommendations, str):
            print(f"❌ {recommendations}")
        else:
            for idx, row in recommendations.iterrows():
                genres_str = ', '.join(row['genres'][:3]) if isinstance(row['genres'], list) else 'N/A'
                overview = row['overview'][:100] + '...' if len(row['overview']) > 100 else row['overview']
                print(f"{idx+1}. {row['title']}")
                print(f"   Genre: {genres_str}")
                print(f"   Rating: {row['vote_average']:.1f}/10")
                print(f"   {overview}\n")
    
    def predict_rating(self):
        """Predict rating for user-movie pair"""
        print("\n" + "-"*70)
        print("RATING PREDICTION")
        print("-"*70)
        
        user_id = int(input("Enter User ID: "))
        movie_id = int(input("Enter Movie ID: "))
        
        predicted = self.recommender.predict_rating(user_id, movie_id)
        
        print(f"\n⭐ Predicted rating: {predicted}/5.0")
    
    def get_personalized_recommendations(self):
        """Get personalized recommendations"""
        print("\n" + "-"*70)
        print("PERSONALIZED RECOMMENDATIONS")
        print("-"*70)
        
        user_id = int(input("Enter User ID: "))
        n = int(input("Number of recommendations (default 10): ") or "10")
        
        print(f"\n🎯 Top {n} recommendations for User {user_id}:\n")
        recommendations = self.recommender.get_top_n_recommendations(
            user_id, n, self.ratings_df
        )
        
        if isinstance(recommendations, str):
            print(f"❌ {recommendations}")
        else:
            for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
                # Find movie details
                movie = self.movies_df[self.movies_df['id'] == movie_id]
                if not movie.empty:
                    title = movie.iloc[0]['title']
                    print(f"{i}. {title} (Movie ID: {movie_id})")
                    print(f"   Predicted Rating: {predicted_rating:.2f}/5.0\n")
    
    def add_rating(self):
        """Add new user rating"""
        print("\n" + "-"*70)
        print("ADD NEW RATING")
        print("-"*70)
        
        user_id = int(input("Enter User ID: "))
        movie_id = int(input("Enter Movie ID: "))
        rating = float(input("Enter Rating (0.5 - 5.0): "))
        
        if not (0.5 <= rating <= 5.0):
            print("❌ Invalid rating! Must be between 0.5 and 5.0")
            return
        
        self.tracker.add_rating(user_id, movie_id, rating)
        print(f"\n✅ Rating added successfully!")
        
        # Check if retraining is needed
        count = self.tracker.get_new_interactions_count()
        threshold = self.scheduler.threshold
        print(f"\n📊 New ratings: {count}/{threshold} (Retraining threshold)")
        
        if count >= threshold:
            print("\n⚠️  Retraining threshold reached!")
            choice = input("Would you like to retrain the model now? (y/n): ")
            if choice.lower() == 'y':
                self.check_retraining()
    
    def show_statistics(self):
        """Show system statistics"""
        print("\n" + "="*70)
        print("SYSTEM STATISTICS")
        print("="*70)
        
        print(f"\n📊 Dataset Information:")
        print(f"   Total Movies: {len(self.movies_df)}")
        print(f"   Total Ratings: {len(self.ratings_df)}")
        print(f"   Total Users: {self.ratings_df['userId'].nunique()}")
        print(f"   Average Rating: {self.ratings_df['rating'].mean():.2f}")
        
        print(f"\n🆕 New Interactions:")
        new_count = self.tracker.get_new_interactions_count()
        print(f"   Pending Ratings: {new_count}")
        print(f"   Retraining Threshold: {self.scheduler.threshold}")
        
        print(f"\n⭐ Top Rated Movies:")
        top_movies = self.movies_df.nlargest(5, 'vote_average')[['title', 'vote_average', 'vote_count']]
        for idx, row in top_movies.iterrows():
            print(f"   • {row['title']}: {row['vote_average']:.1f}/10 ({row['vote_count']} votes)")
        
        # Load training history
        history_file = Path("logs/training_logs/training_history.json")
        if history_file.exists():
            import json
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            print(f"\n🔄 Training History:")
            for i, record in enumerate(history[-5:], 1):  # Show last 5
                print(f"   {i}. {record['timestamp'][:19]}")
                print(f"      RMSE: {record['rmse']:.4f}, MAE: {record['mae']:.4f}")
    
    def check_retraining(self):
        """Check and perform retraining if needed"""
        print("\n" + "-"*70)
        print("RETRAINING CHECK")
        print("-"*70)
        
        new_count = self.tracker.get_new_interactions_count()
        print(f"\n📊 New interactions: {new_count}")
        
        if new_count < self.scheduler.threshold:
            print(f"⏳ Need {self.scheduler.threshold - new_count} more ratings to trigger retraining")
            return
        
        print("\n🔄 Starting retraining process...")
        
        # Ensure data is loaded
        if self.movies_df is None:
            self.load_data()
        
        success = self.scheduler.check_and_retrain(self.movies_df)
        
        if success:
            print("\n✅ Retraining completed successfully!")
            print("🔄 Reloading models...")
            
            # Reload data and models
            self.load_data()
            self.recommender = MovieRecommender()
            
            print("✅ Models reloaded!")
        else:
            print("\n❌ Retraining failed or not needed")
    
    def search_movies(self):
        """Search for movies"""
        print("\n" + "-"*70)
        print("MOVIE SEARCH")
        print("-"*70)
        
        query = input("\nEnter search query: ").strip().lower()
        
        # Search in titles
        mask = self.movies_df['title'].str.lower().str.contains(query, na=False)
        results = self.movies_df[mask][['title', 'genres', 'vote_average', 'popularity']]
        
        if len(results) == 0:
            print(f"\n❌ No movies found matching '{query}'")
        else:
            print(f"\n🔍 Found {len(results)} movies:\n")
            for idx, row in results.head(20).iterrows():
                genres_str = ', '.join(row['genres'][:3]) if isinstance(row['genres'], list) else 'N/A'
                print(f"  • {row['title']}")
                print(f"    Genre: {genres_str} | Rating: {row['vote_average']:.1f}/10\n")
    
    def run(self):
        """Main application loop"""
        # Setup
        if not self.setup():
            return
        
        print("\n✅ System ready!")
        
        # Main loop
        while True:
            try:
                self.show_menu()
                choice = input("\nEnter your choice (1-8): ").strip()
                
                if choice == '1':
                    self.get_recommendations()
                elif choice == '2':
                    self.predict_rating()
                elif choice == '3':
                    self.get_personalized_recommendations()
                elif choice == '4':
                    self.add_rating()
                elif choice == '5':
                    self.show_statistics()
                elif choice == '6':
                    self.check_retraining()
                elif choice == '7':
                    self.search_movies()
                elif choice == '8':
                    print("\n👋 Thank you for using Movie Recommender System!")
                    break
                else:
                    print("\n❌ Invalid choice! Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ An error occurred: {e}")
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║              🎬 MOVIE RECOMMENDATION SYSTEM 🎬                      ║
    ║                                                                      ║
    ║                  MLOps Project with Auto-Retraining                  ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = MovieRecommenderApp()
    app.run()


if __name__ == "__main__":
    main()
