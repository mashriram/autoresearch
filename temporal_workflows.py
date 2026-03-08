from datetime import timedelta
import subprocess
import json
import os
from temporalio import workflow, activity

@activity.defn
async def run_training_trial(params: dict) -> dict:
    """
    Executes a training run on the worker's hardware and returns metrics.
    """
    run_name = params.get("run_name", "temporal_run")
    
    # Securely wipe legacy metrics payload
    if os.path.exists("run_metrics.json"):
        os.remove("run_metrics.json")
    
    # Format CLI parameters appropriately
    cmd = ["uv", "run", "train.py", "--run-name", run_name]
    
    # Map parameter names natively based on standard argparse format
    cmd_flags = {
        "embedding_lr": "--embedding-lr",
        "matrix_lr": "--matrix-lr",
        "depth": "--depth",
        "batch_size": "--batch-size"
    }

    for key, arg in cmd_flags.items():
        if key in params:
            cmd.extend([arg, str(params[key])])

    activity.logger.info(f"Temporal Runner Invoking: {' '.join(cmd)}")

    # Execute deterministic loop process
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        activity.logger.error(f"Training Failed:\n{e.stderr}")
        raise Exception(f"Failed to complete training loop. Return code: {e.returncode}")

    # Recover strictly structured standard interface output
    try:
        with open("run_metrics.json", "r") as f:
            metrics = json.load(f)
            return metrics
    except Exception as e:
        activity.logger.error("Failed parsing valid runtime JSON from script.")
        raise Exception("run_metrics.json not found or malformed.")

@workflow.defn
class DistributedResearchCampaign:
    @workflow.run
    async def run(self, configurations: list[dict]) -> dict:
        """
        Coordinates a multi-hardware evaluation swarm.
        Takes multiple hyperparameter dictionary specs.
        """
        results = []
        for config in configurations:
            # We schedule tasks sequentially here, but you can utilize asyncio.gather
            # for fully parallel multi-agent evaluation trees.
            metrics = await workflow.execute_activity(
                run_training_trial,
                config,
                start_to_close_timeout=timedelta(hours=2) # Highly variable depending on GPU
            )
            results.append({
                "config": config,
                "metrics": metrics
            })
            
        best_run = min(results, key=lambda x: x["metrics"].get("val_bpb", float("inf")))
        return best_run
