import subprocess
import sys
import time
from pathlib import Path
 
 
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline_config import load_config
 
 
PIPELINE_STEPS = [
    (
        "1. Data Ingestion",
        "src/ingestion/ingest.py"
    ),
    (
        "2. Data Validation",
        "src/validation/validate.py"
    ),
    (
        "3. Data Preparation",
        "src/preparation/prepare.py"
    ),
    (
        "4. Data Transformation",
        "src/transformation/transform.py"
    ),
    (
        "5. Feature Engineering",
        "src/features/build_features.py"
    ),
    (
        "6. Popularity Model Training",
        "src/training/popularity_model.py"
    ),
    (
        "7. Collaborative Filtering Training",
        "src/training/collaborative_filtering.py"
    ),
    (
        "8. Model Evaluation",
        "src/evaluation/evaluate.py"
    ),
    (
        "9. MLflow Experiment Tracking",
        "src/evaluation/track_experiments.py"
    ),
]
 
 
def run_step(step_number, step_name, script_path):
 
    print("\n" + "=" * 70)
    print(f"PIPELINE STEP {step_number}")
    print(step_name)
    print("=" * 70)
 
    script = PROJECT_ROOT / script_path
 
    if not script.exists():
        print(f"\nERROR: Script not found: {script}")
        return False
 
    start_time = time.time()
 
    try:
 
        subprocess.run(
            [
                sys.executable,
                str(script)
            ],
            cwd=PROJECT_ROOT,
            check=True
        )
 
        end_time = time.time()
        duration = end_time - start_time
 
        print(
            f"\nSUCCESS: {step_name}"
        )
 
        print(
            f"Execution time: {duration:.2f} seconds"
        )
 
        return True
 
    except subprocess.CalledProcessError as error:
 
        print(
            f"\nFAILED: {step_name}"
        )
 
        print(
            f"Process returned exit code: "
            f"{error.returncode}"
        )
 
        return False
 
    except Exception as error:
 
        print(
            f"\nUNEXPECTED ERROR: {error}"
        )
 
        return False
 
 
def run_pipeline():
    config = load_config()
 
    print("\n" + "=" * 70)
    print("END-TO-END RECOMMENDATION SYSTEM PIPELINE")
    print("=" * 70)
 
    print("\nProject:")
    print("Data Management for Machine Learning - Assignment 2")
 
    print("\nDataset:")
    print(config["dataset"]["name"])
 
    print("\nPipeline execution started.")
 
    pipeline_start = time.time()
 
    completed_steps = []
 
    for number, (name, script) in enumerate(
        PIPELINE_STEPS,
        start=1
    ):
 
        success = run_step(
            number,
            name,
            script
        )
 
        if not success:
 
            print("\n" + "=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)
 
            print(
                f"\nFailure occurred during:"
                f"\n{name}"
            )
 
            print(
                "\nRemaining steps were not executed."
            )
 
            sys.exit(1)
 
        completed_steps.append(name)
 
    pipeline_end = time.time()
 
    total_time = (
        pipeline_end - pipeline_start
    )
 
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
 
    for step in completed_steps:
        print(f"[COMPLETED] {step}")
 
    print(
        f"\nTotal execution time: "
        f"{total_time:.2f} seconds"
    )
 
    print("\nOutputs generated:")
 
    print(
        "- Cleaned datasets: data/processed/"
    )
 
    print(
        "- Feature datasets: data/features/"
    )
 
    print(
        "- Validation report: data/errors/"
    )
 
    print(
        "- Model artifacts: models/"
    )
 
    print(
        "- Feature catalog: reports/feature_catalog.csv"
    )
 
    print(
        "- Evaluation results: reports/model_evaluation.csv"
    )
 
    print(
        "- MLflow tracking database: mlflow.db"
    )
 
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
 
 
if __name__ == "__main__":
    run_pipeline()