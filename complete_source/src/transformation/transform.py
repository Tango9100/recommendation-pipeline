import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
def transform_data():
 
    config = load_config()
    processed_dir = resolve_path(config, "preparation.paths.processed")
    features_dir = resolve_path(config, "features.output_path")
 
    features_dir.mkdir(
        parents=True,
        exist_ok=True
    )
 
    print("=" * 60)
    print("DATA TRANSFORMATION")
    print("=" * 60)
 
    ratings = pd.read_csv(
        processed_dir / "ratings_clean.csv"
    )
 
    movies = pd.read_csv(
        processed_dir / "movies_clean.csv"
    )
 
    ratings["interaction_time"] = pd.to_datetime(
        ratings["interaction_time"]
    )
 
    # ---------------------------------------------
    # 1. User Activity Summary
    # ---------------------------------------------
 
    user_activity = (
        ratings
        .groupby("userId")
        .agg(
            interaction_count=("movieId", "count"),
            unique_movies=("movieId", "nunique"),
            average_rating=("rating", "mean"),
            last_interaction=(
                "interaction_time",
                "max"
            )
        )
        .reset_index()
    )
 
    user_activity[
        "average_rating"
    ] = user_activity[
        "average_rating"
    ].round(3)
 
    # ---------------------------------------------
    # 2. Item Popularity
    # ---------------------------------------------
 
    item_popularity = (
        ratings
        .groupby("movieId")
        .agg(
            rating_count=("userId", "count"),
            unique_users=("userId", "nunique"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )
 
    item_popularity[
        "average_rating"
    ] = item_popularity[
        "average_rating"
    ].round(3)
 
    item_popularity = item_popularity.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )
 
    item_popularity = item_popularity.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )
 
    # ---------------------------------------------
    # 3. User-Item Matrix
    # ---------------------------------------------
 
    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )
 
    # ---------------------------------------------
    # Save outputs
    # ---------------------------------------------
 
    user_activity.to_csv(
        features_dir / "user_activity.csv",
        index=False
    )
 
    item_popularity.to_csv(
        features_dir / "item_popularity.csv",
        index=False
    )
 
    user_item_matrix.to_csv(
        features_dir / "user_item_matrix.csv"
    )
 
    print("\nUser activity summary:")
    print(user_activity.head())
 
    print("\nMost popular movies:")
    print(
        item_popularity[
            [
                "movieId",
                "title",
                "rating_count",
                "average_rating"
            ]
        ].head(10)
    )
 
    print("\nUser-item matrix shape:")
    print(user_item_matrix.shape)
 
    print("\nFiles created:")
 
    print("data/features/user_activity.csv")
    print("data/features/item_popularity.csv")
    print("data/features/user_item_matrix.csv")
 
    print("\nTransformation completed successfully.")
 
    return (
        user_activity,
        item_popularity,
        user_item_matrix
    )
 
 
if __name__ == "__main__":
    transform_data()
