import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
def build_features():
 
    config = load_config()
    processed_dir = resolve_path(config, "preparation.paths.processed")
    features_dir = resolve_path(config, "features.output_path")
    catalog_path = resolve_path(config, "features.catalog_path")
 
    features_dir.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
 
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
 
    # Build serving features from training data to prevent test leakage.
    ratings = pd.read_csv(
        processed_dir / "train.csv"
    )
 
    movies = pd.read_csv(
        processed_dir / "movies_clean.csv"
    )
    users = pd.read_csv(
        processed_dir / "users.csv"
    )
 
    ratings["interaction_time"] = pd.to_datetime(
        ratings["interaction_time"]
    )
 
    # --------------------------------------------------
    # USER FEATURES
    # --------------------------------------------------
 
    user_aggregates = (
        ratings.groupby("userId")
        .agg(
            user_avg_rating=("rating", "mean"),
            user_activity_count=("movieId", "count"),
            user_unique_movies=("movieId", "nunique"),
            last_interaction=("interaction_time", "max")
        )
        .reset_index()
    )
 
    user_features = users.merge(user_aggregates, on="userId", how="left")
    user_features["user_activity_count"] = (
        user_features["user_activity_count"].fillna(0).astype(int)
    )
    user_features["user_unique_movies"] = (
        user_features["user_unique_movies"].fillna(0).astype(int)
    )
    latest_date = ratings["interaction_time"].max()
 
    user_features["days_since_last_interaction"] = (
        latest_date - user_features["last_interaction"]
    ).dt.days
 
    user_features["user_avg_rating"] = (
        user_features["user_avg_rating"].round(3)
    )
 
    # --------------------------------------------------
    # ITEM FEATURES
    # --------------------------------------------------
 
    item_aggregates = (
        ratings.groupby("movieId")
        .agg(
            item_popularity=("userId", "count"),
            item_unique_users=("userId", "nunique"),
            item_avg_rating=("rating", "mean")
        )
        .reset_index()
    )
 
    item_features = movies.merge(item_aggregates, on="movieId", how="left")
    item_features["item_popularity"] = (
        item_features["item_popularity"].fillna(0).astype(int)
    )
    item_features["item_unique_users"] = (
        item_features["item_unique_users"].fillna(0).astype(int)
    )
    item_features["item_avg_rating"] = (
        item_features["item_avg_rating"].round(3)
    )
 
    interaction_features = ratings.copy()
    interaction_features["interaction_hour"] = (
        interaction_features["interaction_time"].dt.hour
    )
    interaction_features["interaction_day_of_week"] = (
        interaction_features["interaction_time"].dt.dayofweek
    )
    interaction_features["interaction_month"] = (
        interaction_features["interaction_time"].dt.month
    )
 
    # --------------------------------------------------
    # SAVE FEATURES
    # --------------------------------------------------
 
    user_features.to_csv(
        features_dir / "user_features.csv",
        index=False
    )
 
    item_features.to_csv(
        features_dir / "item_features.csv",
        index=False
    )
    interaction_features.to_csv(
        features_dir / "interaction_features.csv",
        index=False
    )
 
    # --------------------------------------------------
    # FEATURE CATALOG
    # --------------------------------------------------
 
    catalog = pd.DataFrame([
        {
            "feature": "userId",
            "entity": "User",
            "type": "Integer",
            "description": "MovieLens user identifier; no profile attributes are provided"
        },
        {
            "feature": "user_avg_rating",
            "entity": "User",
            "type": "Numeric",
            "description": "Average rating given by the user"
        },
        {
            "feature": "user_activity_count",
            "entity": "User",
            "type": "Integer",
            "description": "Total number of user interactions"
        },
        {
            "feature": "user_unique_movies",
            "entity": "User",
            "type": "Integer",
            "description": "Number of unique movies rated by user"
        },
        {
            "feature": "days_since_last_interaction",
            "entity": "User",
            "type": "Integer",
            "description": "Days since user's most recent interaction"
        },
        {
            "feature": "movieId",
            "entity": "Movie",
            "type": "Integer",
            "description": "MovieLens item identifier"
        },
        {
            "feature": "item_popularity",
            "entity": "Movie",
            "type": "Integer",
            "description": "Total number of ratings received"
        },
        {
            "feature": "item_unique_users",
            "entity": "Movie",
            "type": "Integer",
            "description": "Number of unique users rating the movie"
        },
        {
            "feature": "item_avg_rating",
            "entity": "Movie",
            "type": "Numeric",
            "description": "Average rating received by movie"
        },
        {
            "feature": "genres",
            "entity": "Movie",
            "type": "Categorical",
            "description": "Movie genre information"
        },
        {
            "feature": "rating",
            "entity": "Interaction",
            "type": "Numeric",
            "description": "Explicit rating supplied by a user for a movie"
        },
        {
            "feature": "interaction_time",
            "entity": "Interaction",
            "type": "Datetime",
            "description": "UTC timestamp converted to a datetime value"
        },
        {
            "feature": "interaction_hour",
            "entity": "Interaction",
            "type": "Integer",
            "description": "Hour of day when the interaction occurred"
        },
        {
            "feature": "interaction_day_of_week",
            "entity": "Interaction",
            "type": "Integer",
            "description": "Day of week when the interaction occurred, Monday=0"
        },
        {
            "feature": "interaction_month",
            "entity": "Interaction",
            "type": "Integer",
            "description": "Month when the interaction occurred"
        }
    ])
 
    catalog.to_csv(
        catalog_path,
        index=False
    )
 
    print("\nUser features created:")
    print(user_features.head())
 
    print("\nItem features created:")
    print(
        item_features[
            [
                "movieId",
                "title",
                "item_popularity",
                "item_avg_rating"
            ]
        ].head()
    )
 
    print("\nFeature Catalog:")
    print(catalog)
 
    print("\nFiles created:")
    print("data/features/user_features.csv")
    print("data/features/item_features.csv")
    print("data/features/interaction_features.csv")
    print("reports/feature_catalog.csv")
 
    print("\nFeature engineering completed successfully.")
 
 
if __name__ == "__main__":
    build_features()