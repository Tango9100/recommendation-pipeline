import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
def train_collaborative_model():
 
    config = load_config()
    processed_dir = resolve_path(config, "preparation.paths.processed")
    features_dir = resolve_path(config, "features.output_path")
    models_dir = resolve_path(config, "models.output_path")
    train_path = features_dir / "interaction_features.csv"
    movies_path = processed_dir / "movies_clean.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
 
    print("=" * 60)
    print("MODEL 2: ITEM-BASED COLLABORATIVE FILTERING")
    print("=" * 60)
 
    train = pd.read_csv(train_path)
    movies = pd.read_csv(movies_path)
 
    print(f"\nTraining interactions: {len(train):,}")
    print(f"Users: {train['userId'].nunique():,}")
    print(f"Movies: {train['movieId'].nunique():,}")
 
    # Create movie-user matrix
    movie_user_matrix = train.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
        fill_value=0
    )
 
    print(
        f"\nMovie-user matrix shape: "
        f"{movie_user_matrix.shape}"
    )
 
    # Convert to sparse matrix
    sparse_matrix = csr_matrix(
        movie_user_matrix.values
    )
 
    # Create nearest-neighbour model using cosine distance
    model = NearestNeighbors(
        metric=config["models"]["collaborative_filtering"]["similarity_metric"],
        algorithm="brute"
    )
 
    model.fit(sparse_matrix)
 
    # Save everything needed for predictions
    model_artifact = {
        "model": model,
        "movie_ids": movie_user_matrix.index.tolist(),
        "user_ids": movie_user_matrix.columns.tolist()
    }
 
    model_path = (
        models_dir / "collaborative_filtering_model.pkl"
    )
 
    with open(model_path, "wb") as file:
        pickle.dump(model_artifact, file)
 
    print("\nCollaborative filtering model trained.")
    print(f"Model saved to: {model_path}")
    print(
        "\nCollaborative filtering training "
        "completed successfully."
    )
 
    return model, movie_user_matrix, train, movies
 
 
def recommend_movies(
    user_id,
    model,
    movie_user_matrix,
    train,
    movies,
    k=5,
    positive_rating_threshold=4.0,
    max_seed_movies=10
):
 
    # Movies rated by this user
    user_history = train[
        train["userId"] == user_id
    ].sort_values(
        "rating",
        ascending=False
    )
 
    if user_history.empty:
        return pd.DataFrame()
 
    # Focus on movies the user positively rated.
    liked_movies = user_history[
        user_history["rating"] >= positive_rating_threshold
    ]
 
    if liked_movies.empty:
        liked_movies = user_history.head(5)
 
    # Limit seeds for efficient recommendation.
    liked_movies = liked_movies.head(max_seed_movies)
 
    recommendation_scores = {}
 
    movie_ids = list(movie_user_matrix.index)
 
    for movie_id in liked_movies["movieId"]:
 
        if movie_id not in movie_user_matrix.index:
            continue
 
        movie_position = movie_ids.index(movie_id)
 
        number_neighbors = min(
            11,
            len(movie_ids)
        )
 
        distances, indices = model.kneighbors(
            movie_user_matrix.iloc[
                movie_position
            ].values.reshape(1, -1),
            n_neighbors=number_neighbors
        )
 
        for distance, index in zip(
            distances.flatten(),
            indices.flatten()
        ):
 
            similar_movie_id = movie_ids[index]
 
            if similar_movie_id == movie_id:
                continue
 
            similarity = 1 - distance
 
            if similar_movie_id not in recommendation_scores:
                recommendation_scores[similar_movie_id] = 0
 
            recommendation_scores[
                similar_movie_id
            ] += similarity
 
    # Remove movies already rated by user
    seen_movies = set(user_history["movieId"])
 
    recommendation_scores = {
        movie_id: score
        for movie_id, score
        in recommendation_scores.items()
        if movie_id not in seen_movies
    }
 
    # Sort recommendations
    ranked = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:k]
 
    recommendations = pd.DataFrame(
        ranked,
        columns=["movieId", "score"]
    )
 
    if recommendations.empty:
        return recommendations
 
    recommendations = recommendations.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )
 
    return recommendations
 
 
if __name__ == "__main__":
    config = load_config()
    features_dir = resolve_path(config, "features.output_path")
    (
        model,
        matrix,
        train,
        movies
    ) = train_collaborative_model()
 
    sample_user = train["userId"].iloc[0]
 
    recommendations = recommend_movies(
        sample_user,
        model,
        matrix,
        train,
        movies,
        k=config["recommendations"]["default_count"],
        positive_rating_threshold=(
            config["models"]["collaborative_filtering"][
                "positive_rating_threshold"
            ]
        ),
        max_seed_movies=config["models"]["collaborative_filtering"][
            "max_seed_movies"
        ]
    )
 
    print(
        f"\nTop 5 personalized recommendations "
        f"for User {sample_user}:\n"
    )
 
    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(
            recommendations[
                [
                    "movieId",
                    "title",
                    "genres",
                    "score"
                ]
            ].to_string(index=False)
        )
 