import pandas as pd
import numpy as np
from ast import literal_eval
import pickle
import os
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Handle all data processing operations"""
    
    def __init__(self, raw_path="data/raw", processed_path="data/processed"):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load raw data files"""
        logger.info("Loading data...")
        
        credits = pd.read_csv(self.raw_path / "tmdb_5000_credits.csv")
        movies = pd.read_csv(self.raw_path / "tmdb_5000_movies.csv")
        ratings = pd.read_csv(self.raw_path / "ratings_small.csv")
        
        return credits, movies, ratings
    
    def process_metadata(self, credits, movies):
        """Process and merge movie metadata"""
        logger.info("Processing metadata...")
        
        # Drop duplicate title from credits
        credits = credits.drop('title', axis=1)
        credits.columns = ['id', 'cast', 'crew']
        
        # Merge
        df = movies.merge(credits, on='id')
        
        # Fill missing values
        df['overview'] = df['overview'].fillna('')
        
        return df
    
    def extract_features(self, df):
        """Extract features from JSON-like strings"""
        logger.info("Extracting features...")
        
        def safe_eval(x):
            try:
                return literal_eval(x) if pd.notna(x) else []
            except:
                return []
        
        # Extract genres
        df['genres'] = df['genres'].apply(
            lambda x: [i['name'] for i in safe_eval(x)]
        )
        
        # Extract keywords
        df['keywords'] = df['keywords'].apply(
            lambda x: [i['name'] for i in safe_eval(x)]
        )
        
        # Extract top 3 cast
        df['cast'] = df['cast'].apply(
            lambda x: [i['name'] for i in safe_eval(x)][:3]
        )
        
        # Extract director
        def get_director(x):
            for i in safe_eval(x):
                if i.get('job') == 'Director':
                    return [i['name']]
            return []
        
        df['crew'] = df['crew'].apply(get_director)
        
        # Clean text - remove spaces
        for col in ['genres', 'keywords', 'cast', 'crew']:
            df[col] = df[col].apply(
                lambda x: [str.lower(i.replace(" ", "")) for i in x]
            )
        
        # Create soup
        df['soup'] = (df['keywords'] + df['cast'] + 
                      df['crew'] + df['genres'])
        df['soup_str'] = df['soup'].apply(lambda x: ' '.join(x))
        
        return df
    
    def save_processed_data(self, movies_df, ratings_df):
        """Save processed data"""
        logger.info("Saving processed data...")
        
        movies_df.to_pickle(self.processed_path / 'movies_processed.pkl')
        ratings_df.to_csv(self.processed_path / 'ratings_processed.csv', index=False)
        
        logger.info("Data saved successfully!")
    
    def run(self):
        """Run complete pipeline"""
        credits, movies, ratings = self.load_data()
        df = self.process_metadata(credits, movies)
        df = self.extract_features(df)
        self.save_processed_data(df, ratings)
        return df, ratings

