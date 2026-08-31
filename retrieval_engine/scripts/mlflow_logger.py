import mlflow
import os

EXPERIMENT_NAME = "retrieval-eval-experiments"
mlflow.set_experiment(EXPERIMENT_NAME)

def start_parent_run(run_name: str = "experiment"):
    """
    Starts a parent MLflow run and stores its run_id in an env var
    so child scripts can attach to it.
    """
    run = mlflow.start_run(run_name=run_name)
    os.environ["PARENT_RUN_ID"] = run.info.run_id
    return run.info.run_id

def start_child_run(run_name: str):
    """
    Starts a nested MLflow run inside the parent run.
    Requires that PARENT_RUN_ID is set in the environment.
    """
    parent_id = os.environ.get("PARENT_RUN_ID")
    if not parent_id:
        RuntimeError("No parent run ID found. Did you forget to start the parent run?")
    return mlflow.start_run(run_name=run_name, nested=True)

def end_run():
    """Ends the currently active MLflow run."""
