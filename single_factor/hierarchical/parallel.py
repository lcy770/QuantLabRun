"""Synchronous on-policy collection in independent CPU processes."""
from concurrent.futures import ProcessPoolExecutor
import random
from time import perf_counter

import torch

from .ast import Grammar, Node
from .evaluation import load_task
from .models import LowerPolicy, UpperPolicy
from .ppo import continuation_returns
from .search import episode


def initialize(config):
    global _config, _grammar, _lower, _upper, _evaluators
    _config = config
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _grammar = Grammar(**config["grammar"])
    _lower = LowerPolicy(_grammar, config["width"], config["corpus"]["noise_steps"],
                         use_task=config["lower"].get("task_context", False),
                         representation=config["lower"].get("representation", "ast"),
                         use_position=config["lower"].get("position_context", False),
                         edit_backend=config["lower"].get("edit_backend", "ast"))
    _upper = UpperPolicy(_grammar, config["width"], use_archive=config["upper"].get("archive_context", False),
                         representation=config["upper"].get("representation", "ast"),
                         cost_context=config["upper"].get("cost_context", False))
    _evaluators = {}


def evaluator(task):
    from .training import evaluator_for
    if task["name"] not in _evaluators:
        panels = load_task(task, _config["target_horizon"], _config["lookback"], _config["future_days"])
        _evaluators[task["name"]] = evaluator_for(_grammar, panels, _config)
    return _evaluators[task["name"]]


def collect_lower(job):
    started = perf_counter()
    _lower.load_state_dict(job["lower"])
    torch.manual_seed(job["seed"])
    rng = random.Random(job["seed"])
    e = evaluator(job["task"])
    calls, requests = e.calls, e.requests
    trajectories, rewards = [], []
    horizon = _config["lower"].get("diffusion_steps", _lower.noise_steps)
    for start in job["starts"]:
        noisy, _ = _lower.corrupt(start, _config["lower"].get("perturbation_steps", horizon), rng)
        final, rows = _lower.sample(noisy, remaining=horizon)
        trajectories.append(rows)
        rewards.append(e.lower(final))
    branches = continuation_returns(_lower, trajectories, rewards, e.lower,
                  samples=_config["lower"]["continuation_samples"], selected=job["selected"], rng=rng)
    return {"trajectories": trajectories, "rewards": rewards, "branches": branches,
            "calls": e.calls - calls, "requests": e.requests - requests, "worker_seconds": perf_counter() - started}


def collect_episode(job):
    _lower.load_state_dict(job["lower"])
    _upper.load_state_dict(job["upper"])
    torch.manual_seed(job["seed"])
    e = evaluator(job["task"]).for_episode(job["initial"])
    before = {k: getattr(e, k) for k in ("calls", "requests", "hits", "seconds", "combination_evaluations", "combination_seconds")}
    result = episode(_lower, _upper, e, job["initial"], job["search"], random.Random(job["seed"]),
                     output=job["trace"], tag=job["tag"], record_lower=job.get("return_lower", False),
                     lower_training=job.get("lower_training"))
    report = result[3]
    for key, attribute in (("evaluator_calls", "calls"), ("evaluation_requests", "requests"),
                           ("evaluation_cache_hits", "hits"), ("evaluation_seconds", "seconds"),
                           ("combination_evaluations", "combination_evaluations"), ("combination_seconds", "combination_seconds")):
        report[key] -= before[attribute]
    report["trace_file"] = str(job["trace"])
    report["task"] = job["task"]["name"]
    report["public_initial_cache_shared"] = True
    report["generated_cache_episode_local"] = True
    return result if job.get("return_lower") else (result[0], [], [], report)


def prepare_signals(job):
    started = perf_counter()
    e = evaluator(job["task"])
    before = e.calls
    for tree in job["initial"]:
        e.get(tree)
    return {"task": job["task"]["name"], "calls": e.calls - before, "seconds": perf_counter() - started}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class Collector:
    def __init__(self, config):
        self.workers = config["rollout_workers"]
        self.executor = ProcessPoolExecutor(max_workers=self.workers, initializer=initialize, initargs=(config,))

    def lower(self, lower, starts, task, settings, rng):
        state = cpu_state(lower)
        horizon = settings.get("diffusion_steps", lower.noise_steps)
        selected = set(rng.sample([(i, j) for i in range(len(starts)) for j in range(horizon - 1)],
                                 min(settings["continuation_states"], len(starts) * (horizon - 1))))
        chunks = [list(range(i, len(starts), self.workers)) for i in range(min(self.workers, len(starts)))]
        jobs = [{"lower": state, "starts": [starts[i] for i in indices], "task": task,
                 "seed": rng.randrange(2**31), "selected": [(local, j) for local, global_index in enumerate(indices)
                                                              for j in range(horizon - 1) if (global_index, j) in selected]}
                for indices in chunks]
        return list(self.executor.map(collect_lower, jobs))

    def episodes(self, jobs):
        return list(self.executor.map(collect_episode, jobs))

    def prepare(self, tasks, initial):
        jobs = [{"task": task, "initial": initial[i::self.workers]} for task in tasks for i in range(self.workers)]
        return list(self.executor.map(prepare_signals, jobs))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
