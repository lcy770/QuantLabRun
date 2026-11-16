"""Choose CPU settings by measured real rollout and exact PPO update throughput."""
import argparse
import copy
import json
from pathlib import Path
import random
from time import perf_counter

import torch

from .hierarchical.__main__ import load_config
from .hierarchical.ast import Node
from .hierarchical.io import save_json, event
from .hierarchical.parallel import Collector
from .hierarchical.ppo import update
from .hierarchical.training import restore_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--diffusion-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.diffusion_steps:
        config["lower"]["diffusion_steps"] = args.diffusion_steps
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lower, _ = restore_models(bundle)
    corpus = [json.loads(s) for s in Path(args.corpus).read_text().splitlines()]
    library = [Node.from_dict(r["ast"]) for r in corpus if r["split"] == "train"]
    task = next(t for t in config["tasks"] if t["role"] == "train")
    results, trajectories = [], []
    for workers in args.workers:
        settings = copy.deepcopy(config)
        settings["rollout_workers"] = workers
        collector = Collector(settings)
        started = perf_counter()
        try:
            collector.lower(lower, library[:16], task, settings["lower"], random.Random(41))
            startup = perf_counter() - started
            rng = random.Random(42)
            starts = [rng.choice(library) for _ in range(32)]
            started = perf_counter()
            collected = collector.lower(lower, starts, task, settings["lower"], rng)
            duration = perf_counter() - started
            row = {"kind": "rollout", "workers": workers, "formulas": 32,
                   "startup_and_warmup_seconds": startup, "seconds": duration,
                   "formulas_per_second": 32 / duration}
            results.append(row)
            event("cpu_benchmark", **row)
            trajectories = [r for group in collected for trajectory in group["trajectories"] for r in trajectory]
        finally:
            collector.close()
    for threads in (1, 2, 4, 8):
        torch.set_num_threads(threads)
        model = copy.deepcopy(lower)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lower"]["learning_rate"])
        started = perf_counter()
        metrics = update(model, optimizer, trajectories, rng=random.Random(42),
                         **{**config["lower"]["ppo"], "batch_size": 32})
        row = {"kind": "ppo_update", "threads": threads, "seconds": perf_counter() - started,
               "samples": len(trajectories), "old_logp_max_error": metrics["old_logp_max_error"]}
        results.append(row)
        event("cpu_benchmark", **row)
    best_workers = min([r for r in results if r["kind"] == "rollout"], key=lambda r: r["seconds"])["workers"]
    best_threads = min([r for r in results if r["kind"] == "ppo_update"], key=lambda r: r["seconds"])["threads"]
    save_json(args.output, {"results": results, "chosen_workers": best_workers, "chosen_threads": best_threads,
                           "hardware_specific": True, "algorithmic_performance_claim": False})
