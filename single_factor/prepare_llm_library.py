"""Generate traceable DeepSeek formulas and validate them on pre-training history."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import time
from urllib.error import HTTPError

import numpy as np
import torch

from .hierarchical.ast import Grammar, Node, walk
from .hierarchical.evaluation import load_task, Evaluator
from .hierarchical.io import append_json, save_json, digest, event
from custom_bt.ai_research.llm_client import create_llm_client


FAMILIES = ("momentum and reversal", "volume and price-volume confirmation", "range and volatility",
            "liquidity and turnover", "time-series rank and relative strength", "correlation and divergence",
            "normalized price location and asymmetry", "interactions between independent economic hypotheses")


def generate(config, output, batches, count, workers):
    output.mkdir(parents=True, exist_ok=True)
    client = create_llm_client({"llm": {"provider": "deepseek", "model": "deepseek-v4-pro",
                              "temperature": .9, "timeout_seconds": 180}})
    settings = {"grammar": config["grammar"], "model": client.model, "batches": batches, "per_batch": count}
    manifest = output / "generation_settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("generation settings changed; use a separate library directory")
    save_json(manifest, settings)

    def request_batch(index):
        destination = output / "responses" / f"batch_{index:03d}.json"
        if destination.exists():
            return
        family = FAMILIES[index % len(FAMILIES)]
        prompt = (f"Propose {count} distinct quantitative stock factors focused on {family}. "
                  f"Batch {index}; explore different economic mechanisms and interactions, not just window variants. "
                  "Return JSON {\"factors\":[{\"expression\":\"...\",\"hypothesis\":\"...\"}]}. "
                  "Use the following exact functional DSL and budgets:\n" + json.dumps(config["grammar"]) +
                  "\nField units: open/close/high/low=price; volume=shares; amount=price*shares; TurnoverRate=dimensionless. "
                  "add/sub/max/min operands must have matching units. div makes ratios. Constants are dimensionless. "
                  "Rolling unary functions take (expression,window); ts_corr/ts_covariance take (left,right,window). "
                  "A window counts as an AST node. At most 15 total nodes and depth8. Use normalized returns, "
                  "relative volume, ranges and economic interactions. Never correlate constants, take rolling "
                  "variance of constants, use sign of always-positive prices, future information, or reproduce "
                  "raw price/amount through identity transformations. All constants/windows must be allowed. "
                  "No performance claims or fabricated IC. Real IC will be measured independently.")
        for retry in range(3):
            try:
                result = client.complete_json(prompt, "You design financial hypotheses, expressed as strict typed AST formulas.")
                if not isinstance(result.parsed, dict) or not isinstance(result.parsed.get("factors"), list):
                    raise ValueError("response must contain a factors list")
                save_json(destination, {"batch": index, "family": family, "prompt": prompt, "response": result.to_dict()})
                event("llm_batch_saved", batch=index, count=len(result.parsed["factors"]), usage=result.usage)
                return
            except Exception as error:
                event("llm_batch_retry", batch=index, retry=retry, error_type=type(error).__name__)
                if retry == 2:
                    raise
                time.sleep((15 * 2 ** retry) if isinstance(error, HTTPError) and error.code == 429 else 2 ** retry)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(request_batch, i) for i in range(batches)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                event("llm_batch_failed", error_type=type(error).__name__)
    grammar, unique = Grammar(**config["grammar"]), {}
    for path in sorted((output / "responses").glob("*.json")):
        response = json.loads(path.read_text(encoding="utf-8"))
        for index, record in enumerate(response["response"]["parsed"]["factors"]):
            try:
                tree = grammar.parse(record["expression"])
                for _, node in walk(tree):
                    if grammar.windowed.get(node.symbol):
                        operands = node.children[:-1]
                        if any(not any(n.symbol in grammar.fields for _, n in walk(child)) for child in operands):
                            raise ValueError("constant rolling operand")
                unique.setdefault(tree.id, {"id": tree.id, "ast": tree.to_dict(), "expression": tree.text(),
                    "hypothesis": record.get("hypothesis", ""), "family": response["family"],
                    "provenance": {"provider": "deepseek", "model": response["response"]["model"],
                                   "response_file": str(path.resolve()), "record_index": index}})
            except (ValueError, KeyError, SyntaxError, TypeError):
                append_json(output / "syntax_rejections.jsonl", {"response": str(path), "record": record})
    candidates = output / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique.values()), encoding="utf-8")
    event("llm_generation_complete", unique_legal_formulas=len(unique), model=client.model)


def bank_task(config):
    # All library screening labels mature in 2022, before any 2023 upper task.
    cache = next(t for t in config["tasks"] if t["role"] == "train")["fit"]["cache"]
    return {"name": "library_screen_2022", "fit": {"cache": cache, "start": "2022-01-04", "end": "2022-04-15"},
            "search": {"cache": cache, "start": "2022-06-01", "end": "2022-10-31"}}


def initialize_validation(config):
    global _evaluation
    torch.set_num_threads(1)
    panels = load_task(bank_task(config), config["target_horizon"], config["lookback"], config["future_days"])
    _evaluation = Evaluator(Grammar(**config["grammar"]), panels["fit"], panels["search"], cache_size=2)


def score_candidate(record):
    evaluator = _evaluation
    signal = evaluator.get(Node.from_dict(record["ast"]))
    m = signal.metrics
    if not m["reliable"]:
        return {**record, "validated": False, "validation": m}
    active = evaluator.search.active
    daily = (signal.search * evaluator.search.y).sum(1) / evaluator.search.count
    folds = [float(daily[indices].mean()) for indices in np.array_split(np.flatnonzero(active), 3)]
    rank = m["direction"] * m["search"]["rank_ic"]
    passed = abs(m["fit"]["ic"]) >= .01 and m["quality"] >= .02 and rank >= .01 and sum(v > 0 for v in folds) >= 2
    return {**record, "validated": bool(passed), "validation": {**m, "fold_directed_ic": folds,
            "directed_rank_ic": rank, "screening_task": "library_screen_2022"}}


def validate(config, output, workers, target):
    started = time.perf_counter()
    dependencies = [Path(__file__), Path(__file__).parent / "hierarchical/evaluation.py",
                    Path(__file__).parent / "hierarchical/ast.py", Path(__file__).parent / "data.py",
                    Path(__file__).parent / "_vendor/alphagen/data/expression.py",
                    Path(__file__).parent / "_vendor/alphagen/utils/correlation.py"]
    spec = {"implementation": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
            "screening": bank_task(config), "candidates_hash": hashlib.sha256((output / "candidates.jsonl").read_bytes()).hexdigest()}
    spec["grammar"] = config["grammar"]
    spec["target_horizon"] = config["target_horizon"]
    cache = Path(bank_task(config)["fit"]["cache"])
    metadata = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    files = ["manifest.json", "dates.npy", "codes.json", *(f"{f}.npy" for f in config["grammar"]["fields"])]
    if metadata.get("label_file"):
        files.append(metadata["label_file"])
    spec["data_hashes"] = {name: hashlib.sha256((cache / name).read_bytes()).hexdigest() for name in files}
    spec_path = output / "validation_settings.json"
    if (output / "ic_validation.jsonl").exists() and read_spec(spec_path) != spec:
        raise ValueError("validation implementation/data specification differs; use a new library directory")
    save_json(spec_path, spec)
    candidates = [json.loads(s) for s in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    path = output / "ic_validation.jsonl"
    completed = {r["id"]: r for r in (json.loads(s) for s in path.read_text(encoding="utf-8").splitlines())} if path.exists() else {}
    pending = [r for r in candidates if r["id"] not in completed]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_validation, initargs=(config,)) as executor:
        for record in executor.map(score_candidate, pending, chunksize=8):
            append_json(path, record)
            completed[record["id"]] = record
            if len(completed) % 64 == 0:
                event("library_ic_progress", evaluated=len(completed), total=len(candidates),
                      ic_passed=sum(r["validated"] for r in completed.values()))
    qualified = sorted([r for r in completed.values() if r["validated"]],
                       key=lambda r: (-r["validation"]["quality"], r["id"]))
    initialize_validation(config)
    evaluator = _evaluation
    selected, signals = [], []
    # Greedy exact daily-signal diversification, with a small matrix per candidate.
    for record in qualified:
        signal = evaluator.get(Node.from_dict(record["ast"])).fit
        if signals:
            previous = np.stack(signals)
            weights = evaluator.fit.weights
            covariance = np.einsum("kds,ds,ds->kd", previous, signal, weights, optimize=True)
            denominator = np.sqrt(np.einsum("kds,kds,ds->kd", previous, previous, weights, optimize=True) *
                                  np.sum(signal ** 2 * weights, axis=1)[None, :])
            rho = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-24)
            if np.max(np.mean(abs(rho[:, evaluator.fit.active]), axis=1)) >= .98:
                continue
        selected.append(record)
        signals.append(signal)
        if len(selected) >= target:
            break
    library = output / "library.jsonl"
    library.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    manifest = {"created_from": "actual DeepSeek API responses", "candidate_count": len(candidates),
                "ic_passed_before_diversification": len(qualified), "library_size": len(selected), "target": target,
                "screening": bank_task(config), "latest_label_year": 2022,
                "latest_screening_label": str(evaluator.search.ends.max()),
                "gates": {"fit_abs_ic": .01, "validation_directed_ic": .02, "validation_directed_rank_ic": .01,
                          "positive_folds": "at least 2 of 3", "coverage": .8, "max_mean_abs_daily_correlation": .98},
                "library_hash": digest(selected), "future_effectiveness_proven": False}
    manifest.update(validation_spec=spec, validation_wall_seconds=time.perf_counter() - started)
    save_json(output / "library_manifest.json", manifest)
    event("library_validated", **manifest)


def read_spec(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--per-batch", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--target", type=int, default=512)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        generate(config, args.output, args.batches, args.per_batch, args.workers)
    else:
        validate(config, args.output, args.workers, args.target)
