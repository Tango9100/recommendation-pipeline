import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path


def validate_data():
    """Validate the ingested MovieLens movies and ratings datasets."""

    config = load_config()
    movies_path = resolve_path(config, "dataset.paths.movies")
    ratings_path = resolve_path(config, "dataset.paths.ratings")
    errors_dir = resolve_path(config, "validation.errors_path")
    errors_dir.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    print("Starting data validation...")

    report = []

    missing_movies = movies.isnull().sum()
    missing_ratings = ratings.isnull().sum()
    report.append(f"Missing values in movies:\n{missing_movies.to_string()}")
    report.append(f"Missing values in ratings:\n{missing_ratings.to_string()}")

    movies_duplicates = movies.duplicated().sum()
    ratings_duplicates = ratings.duplicated().sum()
    report.append(f"Duplicate rows in movies: {movies_duplicates}")
    report.append(f"Duplicate rows in ratings: {ratings_duplicates}")

    duplicate_movies = movies[movies.duplicated(keep=False)]
    duplicate_ratings = ratings[ratings.duplicated(keep=False)]
    if not duplicate_movies.empty:
        duplicate_movies.to_csv(errors_dir / "duplicate_movies.csv", index=False)
    if not duplicate_ratings.empty:
        duplicate_ratings.to_csv(errors_dir / "duplicate_ratings.csv", index=False)

    expected_movies_columns = {"movieId", "title", "genres"}
    expected_ratings_columns = {"userId", "movieId", "rating", "timestamp"}
    movie_columns_ok = expected_movies_columns.issubset(movies.columns)
    ratings_columns_ok = expected_ratings_columns.issubset(ratings.columns)
    report.append(f"Movies columns validation: {movie_columns_ok}")
    report.append(f"Ratings columns validation: {ratings_columns_ok}")

    invalid_movie_types = pd.Series(False, index=movies.index)
    if "movieId" in movies.columns:
        numeric_movie_ids = pd.to_numeric(movies["movieId"], errors="coerce")
        invalid_movie_types |= (
            numeric_movie_ids.isna()
            | numeric_movie_ids.mod(1).ne(0)
        )
    for column in ("title", "genres"):
        if column in movies.columns:
            invalid_movie_types |= movies[column].notna() & ~movies[column].map(
                lambda value: isinstance(value, str)
            )

    invalid_rating_types = pd.Series(False, index=ratings.index)
    for column in ("userId", "movieId"):
        if column in ratings.columns:
            numeric_ids = pd.to_numeric(ratings[column], errors="coerce")
            invalid_rating_types |= (
                numeric_ids.isna()
                | numeric_ids.mod(1).ne(0)
            )
    if "rating" in ratings.columns:
        invalid_rating_types |= pd.to_numeric(
            ratings["rating"], errors="coerce"
        ).isna()
    if "timestamp" in ratings.columns:
        numeric_timestamps = pd.to_numeric(
            ratings["timestamp"], errors="coerce"
        )
        invalid_rating_types |= (
            numeric_timestamps.isna()
            | numeric_timestamps.mod(1).ne(0)
        )

    report.append(
        f"Invalid movie field types: {int(invalid_movie_types.sum())}"
    )
    report.append(
        f"Invalid rating field types: {int(invalid_rating_types.sum())}"
    )
    if invalid_movie_types.any():
        movies.loc[invalid_movie_types].to_csv(
            errors_dir / "invalid_movie_types.csv", index=False
        )
    if invalid_rating_types.any():
        ratings.loc[invalid_rating_types].to_csv(
            errors_dir / "invalid_rating_types.csv", index=False
        )

    rating_min = config["validation"]["rating_min"]
    rating_max = config["validation"]["rating_max"]
    if "rating" in ratings.columns:
        numeric_ratings = pd.to_numeric(ratings["rating"], errors="coerce")
        invalid_ratings = ratings[
            numeric_ratings.isna()
            | (numeric_ratings < rating_min)
            | (numeric_ratings > rating_max)
        ]
        report.append(
            f"Invalid ratings (outside {rating_min}-{rating_max} range or non-numeric): "
            f"{len(invalid_ratings)}"
        )
        if not invalid_ratings.empty:
            invalid_ratings.to_csv(errors_dir / "invalid_ratings.csv", index=False)
    else:
        report.append("Rating range validation skipped: missing 'rating' column.")

    if "movieId" in ratings.columns and "movieId" in movies.columns:
        valid_movie_ids = pd.to_numeric(movies["movieId"], errors="coerce")
        rating_movie_ids = pd.to_numeric(ratings["movieId"], errors="coerce")
        invalid_movie_refs = ratings[
            ~rating_movie_ids.isin(valid_movie_ids.dropna())
        ]
        report.append(
            f"Invalid movie references in ratings: {len(invalid_movie_refs)}"
        )
        if not invalid_movie_refs.empty:
            invalid_movie_refs.to_csv(
                errors_dir / "invalid_movie_references.csv", index=False
            )
    else:
        report.append(
            "Movie reference validation skipped: missing 'movieId' column."
        )

    if "timestamp" in ratings.columns:
        numeric_timestamps = pd.to_numeric(ratings["timestamp"], errors="coerce")
        convertible_timestamps = pd.to_datetime(
            numeric_timestamps, unit="s", errors="coerce"
        )
        invalid_timestamps = ratings[convertible_timestamps.isna()]
        report.append(f"Invalid timestamps in ratings: {len(invalid_timestamps)}")
        if not invalid_timestamps.empty:
            invalid_timestamps.to_csv(
                errors_dir / "invalid_timestamps.csv", index=False
            )
    else:
        report.append("Timestamp validation skipped: missing 'timestamp' column.")

    report_path = errors_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("MOVIELENS DATA VALIDATION REPORT\n")
        file.write("=" * 40 + "\n\n")
        file.write("\n\n".join(report) + "\n")

    print("\nDATA VALIDATION RESULTS")
    print("=" * 40)
    for line in report:
        print(line)
    print("\nData validation completed successfully!")
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    validate_data()
