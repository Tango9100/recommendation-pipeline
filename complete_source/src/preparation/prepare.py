import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
def prepare_data():
    """
    Clean MovieLens data and create a time-based
    train/test split.
    """
 
    config = load_config()
    movies_path = resolve_path(config, "dataset.paths.movies")
    ratings_path = resolve_path(config, "dataset.paths.ratings")
    processed_dir = resolve_path(config, "preparation.paths.processed")
 
    processed_dir.mkdir(parents=True, exist_ok=True)
 
    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
 
    # -------------------------------------------------
    # 1. Load raw data
    # -------------------------------------------------
 
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
 
    print("\nRaw data loaded successfully.")
    print(f"Movies records : {len(movies):,}")
    print(f"Ratings records: {len(ratings):,}")
 
    # -------------------------------------------------
    # 2. Remove duplicates
    # -------------------------------------------------
 
    movies_before = len(movies)
    ratings_before = len(ratings)
 
    movies = movies.drop_duplicates().copy()
    ratings = ratings.drop_duplicates().copy()
 
    print("\nDuplicate removal:")
    print(
        f"Movies duplicates removed : "
        f"{movies_before - len(movies)}"
    )
    print(
        f"Ratings duplicates removed: "
        f"{ratings_before - len(ratings)}"
    )
 
    # -------------------------------------------------
    # 3. Handle missing values
    # -------------------------------------------------
 
    # IDs are essential and cannot be safely reconstructed.
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId"])
    movies = movies[movies["movieId"].mod(1).eq(0)].copy()

    for column in ("userId", "movieId", "rating", "timestamp"):
        ratings[column] = pd.to_numeric(ratings[column], errors="coerce")
    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating", "timestamp"]
    )
    ratings = ratings[
        ratings["userId"].mod(1).eq(0)
        & ratings["movieId"].mod(1).eq(0)
        & ratings["timestamp"].mod(1).eq(0)
    ].copy()
 
    # Fill non-critical movie metadata.
    movies["title"] = movies["title"].fillna(
        "Unknown Title"
    )
 
    movies["genres"] = movies["genres"].fillna(
        "(no genres listed)"
    )
 
    # -------------------------------------------------
    # 4. Standardize data types
    # -------------------------------------------------
 
    movies["movieId"] = movies["movieId"].astype(int)
 
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
 
    # -------------------------------------------------
    # 5. Keep valid ratings
    # -------------------------------------------------
 
    ratings = ratings[
        ratings["rating"].between(
            config["validation"]["rating_min"],
            config["validation"]["rating_max"]
        )
    ].copy()
 
    # -------------------------------------------------
    # 6. Ensure referential integrity
    # -------------------------------------------------
 
    ratings = ratings[
        ratings["movieId"].isin(movies["movieId"])
    ].copy()
 
    # -------------------------------------------------
    # 7. Convert timestamp to datetime
    # -------------------------------------------------
 
    ratings["interaction_time"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s",
        errors="coerce"
    )
 
    # Remove timestamp records that cannot be interpreted.
    ratings = ratings.dropna(
        subset=["interaction_time"]
    )
 
    # -------------------------------------------------
    # 8. Sort chronologically
    # -------------------------------------------------
 
    ratings = ratings.sort_values(
        ["userId", "interaction_time"]
    ).reset_index(drop=True)
 
    # -------------------------------------------------
    # 9. Time-based train/test split
    # -------------------------------------------------
    #
    # For each user:
    # Latest interaction = TEST
    # Previous interactions = TRAIN
    #
    # This avoids using future interactions when
    # training the recommendation model.
    # -------------------------------------------------
 
    if config["preparation"]["split_strategy"] != "latest_interaction_per_user":
        raise ValueError(
            "Unsupported split strategy: "
            f"{config['preparation']['split_strategy']}"
        )

    test_indices = (
        ratings
        .groupby("userId")["interaction_time"]
        .idxmax()
    )
 
    test = ratings.loc[test_indices].copy()
 
    train = ratings.drop(
        index=test_indices
    ).copy()
 
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    users = ratings[["userId"]].drop_duplicates().sort_values("userId")
 
    # -------------------------------------------------
    # 10. Save processed datasets
    # -------------------------------------------------
 
    movies.to_csv(
        processed_dir / "movies_clean.csv",
        index=False
    )
 
    ratings.to_csv(
        processed_dir / "ratings_clean.csv",
        index=False
    )
 
    train.to_csv(
        processed_dir / "train.csv",
        index=False
    )
 
    test.to_csv(
        processed_dir / "test.csv",
        index=False
    )
    users.to_csv(
        processed_dir / "users.csv",
        index=False
    )
 
    # -------------------------------------------------
    # 11. Display summary
    # -------------------------------------------------
 
    print("\n" + "=" * 60)
    print("PREPARATION RESULTS")
    print("=" * 60)
 
    print(f"\nClean movies           : {len(movies):,}")
    print(f"Clean interactions     : {len(ratings):,}")
    print(f"Training interactions  : {len(train):,}")
    print(f"Testing interactions   : {len(test):,}")
 
    print(
        f"Unique users           : "
        f"{ratings['userId'].nunique():,}"
    )
 
    print(
        f"Unique movies rated    : "
        f"{ratings['movieId'].nunique():,}"
    )
 
    print(
        f"Minimum rating         : "
        f"{ratings['rating'].min()}"
    )
 
    print(
        f"Maximum rating         : "
        f"{ratings['rating'].max()}"
    )
 
    print(
        f"First interaction      : "
        f"{ratings['interaction_time'].min()}"
    )
 
    print(
        f"Last interaction       : "
        f"{ratings['interaction_time'].max()}"
    )
 
    print("\nFiles created:")
 
    print(
        "data/processed/movies_clean.csv"
    )
    print(
        "data/processed/ratings_clean.csv"
    )
    print(
        "data/processed/train.csv"
    )
    print(
        "data/processed/test.csv"
    )
    print(
        "data/processed/users.csv"
    )
 
    print("\nData preparation completed successfully.")
 
    return movies, ratings, train, test
 
 
if __name__ == "__main__":
    prepare_data()
 