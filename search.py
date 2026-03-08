import optuna
import subprocess
import json
import os
import time

# MLFlow runs will be tracked natively inside train.py.
# This script focuses solely on orchestrating Optuna search sweeps.

def objective(trial):
    # Propose hyperparameters using TPE sampling
    embedding_lr_val = trial.suggest_float("embedding_lr", 0.01, 1.0, log=True)
    matrix_lr_val = trial.suggest_float("matrix_lr", 0.005, 0.2, log=True)
    depth_val = trial.suggest_int("depth", 4, 12, step=2)
    batch_size_val = trial.suggest_categorical("batch_size", [2**18, 2**19, 2**20])

    run_name = f"optuna_trial_{trial.number}"
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Hyperparameters: LR(Emb)={embedding_lr_val:.4f}, LR(Mat)={matrix_lr_val:.4f}, Depth={depth_val}, BatchSize={batch_size_val}")

    # Explicitly clear out old JSON artifacts to avoid ghost reads
    if os.path.exists("run_metrics.json"):
        os.remove("run_metrics.json")

    # Construct execution command
    cmd = [
        "uv", "run", "train.py",
        "--embedding-lr", str(embedding_lr_val),
        "--matrix-lr", str(matrix_lr_val),
        "--depth", str(depth_val),
        "--batch-size", str(batch_size_val),
        "--run-name", run_name
    ]

    # Run the compiled train.py
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} crashed during execution. Discarding parameters.")
        raise optuna.exceptions.TrialPruned()

    # Read back evaluation metrics directly from the established JSON contract
    if not os.path.exists("run_metrics.json"):
        print("run_metrics.json not emitted. Treating trial run as Pruned.")
        raise optuna.exceptions.TrialPruned()

    try:
        with open("run_metrics.json", "r") as f:
            metrics = json.load(f)
            val_bpb = metrics.get("val_bpb", 10.0)
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} finished with Val BPB: {val_bpb}")
    return val_bpb

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="autoresearch_optimization",
        direction="minimize",
        storage="sqlite:///optuna_study.db", # Persist results locally
        load_if_exists=True
    )

    print("--- Starting Single-Box Hyperparameter Sweep with Optuna ---")
    study.optimize(objective, n_trials=50)

    print("\n--- Optimization Finished ---")
    print(f"Best Configuration: {study.best_params}")
    print(f"Best Validation BPB: {study.best_value}")
