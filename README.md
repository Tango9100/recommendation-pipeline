# End-to-End Movie Recommendation Pipeline

This project implements the Data Management for Machine Learning Assignment 2 pipeline using the MovieLens Latest Small dataset.

## Setup

Use Python 3.10 or later. The README itself is documentation, not a command to execute. From the project root, run these commands in PowerShell to create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The raw dataset must be available at:

- `data/raw/movies.csv` (`movieId`, `title`, `genres`)
- `data/raw/ratings.csv` (`userId`, `movieId`, `rating`, `timestamp`)

## Run the pipeline

From the project root, use the interpreter path matching the virtual environment you created. The current workspace already has an environment named `venv`:

```powershell
.\venv\Scripts\python.exe run_pipeline.py
```

For a fresh checkout, follow Setup to create `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

The runner executes each stage in dependency order and stops with a non-zero exit code if a stage fails. The pipeline reads settings from `config/config.yml`; paths in that file are relative to the project root.

## Pipeline stages

1. Ingest the MovieLens CSV files.
2. Validate columns, missing values, duplicates, field types, rating bounds, timestamps, and item references. Invalid rows are written under `data/errors/`.
3. Clean and standardize the data, create a per-user latest-interaction test split, and save the user ID dimension table.
4. Create user activity, item popularity, and user-item matrix datasets.
5. Generate user, item, and interaction feature tables. Training features are built from the training split to avoid test leakage.
6. Train popularity and item-based collaborative-filtering recommenders.
7. Compare both models using Precision@K, Recall@K, NDCG@K, Coverage, and Diversity.
8. Log evaluation parameters, metrics, and artifacts to MLflow.

## Outputs

- Cleaned data and split: `data/processed/`
- Derived datasets and reusable features: `data/features/`
- Invalid records and validation report: `data/errors/`
- Trained models: `models/`
- Feature catalog, evaluation comparison, and final report: `reports/`
- MLflow tracking database: `mlflow.db`

Open the MLflow UI from the project root using the MLflow executable in the same environment used to run the pipeline. In the current workspace:

```powershell
.\venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

For a fresh checkout after following Setup, use:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```

## Configuration

Edit `config/config.yml` to change dataset paths, rating limits, the train/test split strategy, popularity threshold, similarity metric, recommendation count, evaluation K, and MLflow settings. Unsupported split strategies are rejected rather than silently ignored.

## Data and limitations

MovieLens Latest Small provides user IDs, movie metadata, and explicit ratings, but no user demographics. `data/processed/users.csv` therefore contains user IDs only; user activity features are derived from training interactions. The recommenders rank items and do not predict rating values, so RMSE and MAE are not applicable. Results may vary if the input dataset changes.

## Assignment artifacts

- [Architecture diagram](reports/pipeline_architecture.svg)
- [Feature catalog](reports/feature_catalog.csv)
- [Evaluation results](reports/model_evaluation.csv)
- [Final report PDF](reports/final_report.pdf)
- [Pipeline run image](reports/pipeline_success.PNG)
