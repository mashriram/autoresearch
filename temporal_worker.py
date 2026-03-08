import asyncio
import argparse

from temporalio.client import Client
from temporalio.worker import Worker

from temporal_workflows import run_training_trial, DistributedResearchCampaign

async def main():
    parser = argparse.ArgumentParser(description="Start a Temporal Worker Node")
    parser.add_argument("--server", type=str, default="localhost:7233", help="Temporal Server Node Address")
    parser.add_argument("--queue", type=str, default="gpu-research-queue", help="Task Queue Binding Name")
    args = parser.parse_args()

    # Create client connected to server at the given address
    client = await Client.connect(args.server)

    # Run the worker to process activities from the queue
    worker = Worker(
        client,
        task_queue=args.queue,
        workflows=[DistributedResearchCampaign],
        activities=[run_training_trial],
    )
    print(f"🚀 Temporal Research Worker Node Active - Listening on {args.queue}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
