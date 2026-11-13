import pandas as pd
import mlflow
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_config import load_config, resolve_path
 
def track_experiments():
 
    config = load_config()
    results_path = resolve_path(config, "evaluation.report_path")
    reports_dir = results_path.parent
    models_dir = resolve_path(config, "models.output_path")
 
    if not results_path.exists():
        raise FileNotFoundError(
            "model_evaluation.csv not found. "
            "Run evaluate.py first."
        )
 
    results = pd.read_csv(results_path)
 
    # Keep MLflow tracking data in a supported, project-local SQLite backend.
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
 
    print("=" * 60)
    print("MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)
 
    print(f"\nExperiment: {experiment_name}")
 
    for _, row in results.iterrows():
 
        model_name = row["Model"]
 
        # MLflow run names should remain filename friendly.
        run_name = str(model_name).replace(" ", "_")
 
        with mlflow.start_run(
            run_name=run_name
        ):
 
            # ------------------------------------------
            # PARAMETERS
            # ------------------------------------------
 
            mlflow.log_param(
                "model_type",
                model_name
            )
 
            mlflow.log_param(
                "K",
                int(row["K"])
            )
 
            mlflow.log_param(
                "evaluated_users",
                int(row["Evaluated Users"])
            )
 
            mlflow.log_param(
                "dataset",
                config["dataset"]["name"]
            )
 
            mlflow.log_param(
                "split_strategy",
                config["preparation"]["split_strategy"]
            )
 
            # ------------------------------------------
            # METRICS
            # ------------------------------------------
 
            k = int(row["K"])
            mlflow.log_metric(
                f"precision_at_{k}",
                float(row[f"Precision@{k}"])
            )

            mlflow.log_metric(
                f"recall_at_{k}",
                float(row[f"Recall@{k}"])
            )

            mlflow.log_metric(
                f"ndcg_at_{k}",
                float(row[f"NDCG@{k}"])
            )
 
            mlflow.log_metric(
                "coverage",
                float(row["Coverage"])
            )
 
            mlflow.log_metric(
                "diversity",
                float(row["Diversity"])
            )
 
            # ------------------------------------------
            # ARTIFACTS
            # ------------------------------------------
 
            mlflow.log_artifact(
                str(results_path),
                artifact_path="evaluation"
            )
 
            feature_catalog = resolve_path(
                config, "features.catalog_path"
            )
 
            if feature_catalog.exists():
 
                mlflow.log_artifact(
                    str(feature_catalog),
                    artifact_path="features"
                )
 
            if model_name == "Popularity":
 
                model_file = (
                    models_dir /
                    "popularity_model.pkl"
                )
 
            else:
 
                model_file = (
                    models_dir /
                    "collaborative_filtering_model.pkl"
                )
 
            if model_file.exists():
 
                mlflow.log_artifact(
                    str(model_file),
                    artifact_path="models"
                )
 
            print(
                f"\nLogged MLflow run: {model_name}"
            )
 
            print(
                f" Precision@{k}: "
                f"{row[f'Precision@{k}']}"
            )
 
            print(
                f" Recall@{k}: "
                f"{row[f'Recall@{k}']}"
            )
 
            print(
                f" NDCG@{k}: "
                f"{row[f'NDCG@{k}']}"
            )
 
            print(
                f" Coverage: "
                f"{row['Coverage']}"
            )
 
            print(
                f" Diversity: "
                f"{row['Diversity']}"
            )
 
    print(
        "\nMLflow experiment tracking "
        "completed successfully."
    )
 
 
if __name__ == "__main__":
    track_experiments()
    