import pandas as pd
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
def train_popularity_model():
 
    config = load_config()
    features_dir = resolve_path(config, "features.output_path")
    models_dir = resolve_path(config, "models.output_path")
    train_path = features_dir / "interaction_features.csv"
    item_features_path = features_dir / "item_features.csv"
 
    models_dir.mkdir(parents=True, exist_ok=True)
 
    print("=" * 60)
    print("MODEL 1: POPULARITY-BASED RECOMMENDER")
    print("=" * 60)
 
    # Load training data
    train = pd.read_csv(train_path)
    item_features = pd.read_csv(item_features_path)
 
    print(f"\nTraining interactions: {len(train):,}")
 
    # Calculate popularity statistics
    popularity = item_features.rename(
        columns={
            "item_popularity": "rating_count",
            "item_avg_rating": "average_rating"
        }
    )
 
    # Avoid movies with only a few ratings
    min_ratings = config["models"]["popularity"]["min_ratings"]
 
    eligible = popularity[
        popularity["rating_count"] >= min_ratings
    ].copy()
 
    # Popularity score:
    # Primarily based on interaction volume.
    eligible = eligible.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )
 
    # Save model artifact
    model_path = models_dir / "popularity_model.pkl"
 
    with open(model_path, "wb") as file:
        pickle.dump(eligible, file)
 
    print("\nTop 10 Popular Movies:\n")
 
    print(
        eligible[
            [
                "movieId",
                "title",
                "rating_count",
                "average_rating"
            ]
        ].head(10).to_string(index=False)
    )
 
    print(
        f"\nMovies eligible for recommendation: "
        f"{len(eligible):,}"
    )
 
    print(
        f"\nModel saved to: {model_path}"
    )
 
    print(
        "\nPopularity model training completed successfully."
    )
 
    return eligible
 
 
def recommend_popular_movies(user_id, popularity, train, k=5):
 
    # Movies already seen by the user
    seen_movies = set(
        train.loc[
            train["userId"] == user_id,
            "movieId"
        ]
    )
 
    # Remove previously seen movies
    recommendations = popularity[
        ~popularity["movieId"].isin(seen_movies)
    ]
 
    return recommendations.head(k)
 
 
if __name__ == "__main__":
    config = load_config()
    features_dir = resolve_path(config, "features.output_path")
    train = pd.read_csv(
        features_dir / "interaction_features.csv"
    )
 
    popularity = train_popularity_model()
 
    # Example recommendation
    sample_user = train["userId"].iloc[0]
 
    recommendations = recommend_popular_movies(
        sample_user,
        popularity,
        train,
        k=config["recommendations"]["default_count"]
    )
 
    print(
        f"\nTop 5 recommendations for User {sample_user}:\n"
    )
 
    print(
        recommendations[
            ["movieId", "title", "rating_count", "average_rating"]
        ].to_string(index=False)
    )
 