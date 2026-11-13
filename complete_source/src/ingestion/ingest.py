import pandas as pd
from pathlib import Path
import sys
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
def ingest_data():
    """Load MovieLens movies and ratings datasets."""
 
    config = load_config()
    movies_path = resolve_path(config, "dataset.paths.movies")
    ratings_path = resolve_path(config, "dataset.paths.ratings")
 
    print("Starting data ingestion...")
 
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
 
    print("\nData ingestion completed successfully!")
 
    print("\nMovies dataset:")
    print(f"Number of rows: {len(movies)}")
    print(f"Columns: {movies.columns.tolist()}")
 
    print("\nRatings dataset:")
    print(f"Number of rows: {len(ratings)}")
    print(f"Columns: {ratings.columns.tolist()}")
 
    print("\nSample movie records:")
    print(movies.head())
 
    print("\nSample rating records:")
    print(ratings.head())
 
    return movies, ratings
 
 
if __name__ == "__main__":
    ingest_data()