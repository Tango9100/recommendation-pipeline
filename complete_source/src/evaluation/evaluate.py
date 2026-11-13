import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
 
# -------------------------------------------------------
# Ranking Metrics
# -------------------------------------------------------
 
def precision_at_k(recommended, relevant, k=5):
    recommended = recommended[:k]
 
    if len(recommended) == 0:
        return 0.0
 
    hits = len(set(recommended) & set(relevant))
 
    return hits / k
 
 
def recall_at_k(recommended, relevant, k=5):
 
    if len(relevant) == 0:
        return 0.0
 
    recommended = recommended[:k]
 
    hits = len(set(recommended) & set(relevant))
 
    return hits / len(relevant)
 
 
def ndcg_at_k(recommended, relevant, k=5):
 
    recommended = recommended[:k]
 
    dcg = 0.0
 
    for i, movie_id in enumerate(recommended):
 
        if movie_id in relevant:
            dcg += 1 / np.log2(i + 2)
 
    ideal_hits = min(len(relevant), k)
 
    if ideal_hits == 0:
        return 0.0
 
    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )
 
    return dcg / idcg
 
 
# -------------------------------------------------------
# Popularity Model
# -------------------------------------------------------
 
def create_popularity_ranking(train, min_ratings=5):
 
    popularity = (
        train.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )
 
    popularity = popularity[
        popularity["rating_count"] >= min_ratings
    ]
 
    popularity = popularity.sort_values(
        ["rating_count", "average_rating"],
        ascending=[False, False]
    )
 
    return popularity["movieId"].tolist()
 
 
def popularity_recommendations(
    user_id,
    train,
    popularity_ranking,
    k=5
):
 
    seen_movies = set(
        train.loc[
            train["userId"] == user_id,
            "movieId"
        ]
    )
 
    recs = [
        movie_id
        for movie_id in popularity_ranking
        if movie_id not in seen_movies
    ]
 
    return recs[:k]
 
 
# -------------------------------------------------------
# Personalized Collaborative Filtering
# -------------------------------------------------------
 
def build_item_similarity(train):
 
    matrix = train.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
        fill_value=0
    )
 
    similarity = cosine_similarity(matrix.values)
 
    similarity_df = pd.DataFrame(
        similarity,
        index=matrix.index,
        columns=matrix.index
    )
 
    return similarity_df
 
 
def collaborative_recommendations(
    user_id,
    train,
    similarity_df,
    k=5,
    positive_rating_threshold=4.0,
    max_seed_movies=10
):
 
    history = train[
        train["userId"] == user_id
    ].sort_values(
        "rating",
        ascending=False
    )
 
    if history.empty:
        return []
 
    liked = history[
        history["rating"] >= positive_rating_threshold
    ]
 
    if liked.empty:
        liked = history.head(5)
 
    liked = liked.head(max_seed_movies)
 
    scores = {}
 
    for movie_id in liked["movieId"]:
 
        if movie_id not in similarity_df.index:
            continue
 
        similar_movies = (
            similarity_df.loc[movie_id]
            .sort_values(ascending=False)
            .head(20)
        )
 
        for similar_movie, similarity in similar_movies.items():
 
            if similar_movie == movie_id:
                continue
 
            scores[similar_movie] = (
                scores.get(similar_movie, 0)
                + similarity
            )
 
    seen_movies = set(history["movieId"])
 
    scores = {
        movie_id: score
        for movie_id, score in scores.items()
        if movie_id not in seen_movies
    }
 
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
 
    return [
        movie_id
        for movie_id, score in ranked[:k]
    ]
 
 
# -------------------------------------------------------
# Diversity
# -------------------------------------------------------
 
def genre_set(movie_id, movie_genres):
 
    value = movie_genres.get(movie_id, "")
 
    if pd.isna(value):
        return set()
 
    return set(str(value).split("|"))
 
 
def recommendation_diversity(
    recommendations,
    movie_genres
):
 
    if len(recommendations) < 2:
        return 0.0
 
    dissimilarities = []
 
    for i in range(len(recommendations)):
 
        for j in range(i + 1, len(recommendations)):
 
            genres_a = genre_set(
                recommendations[i],
                movie_genres
            )
 
            genres_b = genre_set(
                recommendations[j],
                movie_genres
            )
 
            union = genres_a | genres_b
 
            if len(union) == 0:
                continue
 
            intersection = genres_a & genres_b
 
            similarity = (
                len(intersection) / len(union)
            )
 
            dissimilarity = 1 - similarity
 
            dissimilarities.append(
                dissimilarity
            )
 
    if not dissimilarities:
        return 0.0
 
    return float(
        np.mean(dissimilarities)
    )
 
 
# -------------------------------------------------------
# Evaluate One Model
# -------------------------------------------------------
 
def evaluate_model(
    model_name,
    recommender,
    users,
    train,
    test,
    movie_genres,
    catalog_size,
    k
):
 
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    diversity_scores = []
 
    unique_recommended_items = set()
 
    evaluated_users = 0
 
    for user_id in users:
 
        relevant = test.loc[
            test["userId"] == user_id,
            "movieId"
        ].tolist()
 
        if len(relevant) == 0:
            continue
 
        recommendations = recommender(
            user_id
        )
 
        precision_scores.append(
            precision_at_k(
                recommendations,
                relevant,
                k
            )
        )
 
        recall_scores.append(
            recall_at_k(
                recommendations,
                relevant,
                k
            )
        )
 
        ndcg_scores.append(
            ndcg_at_k(
                recommendations,
                relevant,
                k
            )
        )
 
        diversity_scores.append(
            recommendation_diversity(
                recommendations,
                movie_genres
            )
        )
 
        unique_recommended_items.update(
            recommendations
        )
 
        evaluated_users += 1
 
    coverage = (
        len(unique_recommended_items)
        / catalog_size
        if catalog_size > 0
        else 0
    )
 
    return {
        "Model": model_name,
        "K": k,
        "Evaluated Users": evaluated_users,
        f"Precision@{k}": np.mean(
            precision_scores
        ) if precision_scores else 0,
        f"Recall@{k}": np.mean(
            recall_scores
        ) if recall_scores else 0,
        f"NDCG@{k}": np.mean(
            ndcg_scores
        ) if ndcg_scores else 0,
        "Coverage": coverage,
        "Diversity": np.mean(
            diversity_scores
        ) if diversity_scores else 0
    }
 
 
# -------------------------------------------------------
# Main Evaluation
# -------------------------------------------------------
 
def evaluate_models():
    config = load_config()
    processed_dir = resolve_path(config, "preparation.paths.processed")
    features_dir = resolve_path(config, "features.output_path")
    output_path = resolve_path(config, "evaluation.report_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = config["evaluation"]["k"]
    min_ratings = config["models"]["popularity"]["min_ratings"]
    cf_config = config["models"]["collaborative_filtering"]
 
    print("=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)
 
    train = pd.read_csv(
        features_dir / "interaction_features.csv"
    )
 
    test = pd.read_csv(
        processed_dir / "test.csv"
    )
 
    movies = pd.read_csv(
        processed_dir / "movies_clean.csv"
    )
 
    movie_genres = dict(
        zip(
            movies["movieId"],
            movies["genres"]
        )
    )
 
    catalog_size = len(
        movies[
            movies["movieId"].isin(
                train["movieId"].unique()
            )
        ]
    )
 
    users = sorted(
        set(train["userId"])
        & set(test["userId"])
    )
 
    print(
        f"\nUsers eligible for evaluation: "
        f"{len(users):,}"
    )
 
    print(
        f"Training interactions: "
        f"{len(train):,}"
    )
 
    print(
        f"Test interactions: "
        f"{len(test):,}"
    )
 
    print(
        f"Training catalog size: "
        f"{catalog_size:,}"
    )
 
    # ---------------------------------------------------
    # Popularity model
    # ---------------------------------------------------
 
    print(
        "\nEvaluating popularity model..."
    )
 
    popularity_ranking = create_popularity_ranking(train, min_ratings)
 
    popularity_function = lambda user_id: (
        popularity_recommendations(
            user_id,
            train,
            popularity_ranking,
            k
        )
    )
 
    popularity_results = evaluate_model(
        "Popularity",
        popularity_function,
        users,
        train,
        test,
        movie_genres,
        catalog_size,
        k
    )
 
    # ---------------------------------------------------
    # Collaborative Filtering
    # ---------------------------------------------------
 
    print(
        "Building collaborative filtering "
        "similarity matrix..."
    )
 
    similarity_df = build_item_similarity(
        train
    )
 
    print(
        "Evaluating collaborative filtering model..."
    )
 
    collaborative_function = lambda user_id: (
        collaborative_recommendations(
            user_id,
            train,
            similarity_df,
            k,
            cf_config["positive_rating_threshold"],
            cf_config["max_seed_movies"]
        )
    )
 
    collaborative_results = evaluate_model(
        "Collaborative Filtering",
        collaborative_function,
        users,
        train,
        test,
        movie_genres,
        catalog_size,
        k
    )
 
    # ---------------------------------------------------
    # Results
    # ---------------------------------------------------
 
    results = pd.DataFrame(
        [
            popularity_results,
            collaborative_results
        ]
    )
 
    metric_columns = [
        f"Precision@{k}",
        f"Recall@{k}",
        f"NDCG@{k}",
        "Coverage",
        "Diversity"
    ]
 
    results[metric_columns] = (
        results[metric_columns]
        .round(4)
    )
 
    results.to_csv(
        output_path,
        index=False
    )
 
    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
 
    print(
        results.to_string(
            index=False
        )
    )
 
    print(
        f"\nResults saved to: "
        f"{output_path}"
    )
 
    print(
        "\nModel evaluation "
        "completed successfully."
    )
 
 
if __name__ == "__main__":
    evaluate_models()
 