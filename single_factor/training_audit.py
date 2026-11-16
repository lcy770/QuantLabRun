"""Observe an existing hierarchy run without changing its code, parameters or RNG."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def rows(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may currently be appending its last line; original is kept.
            if index != len(lines) - 1:
                raise ValueError(f"invalid JSON in {path}, line {index + 1}")
    return result


def now():
    return datetime.now().astimezone().isoformat()


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def mean(values):
    return statistics.mean(values) if values else None


def trend(records, key, count):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    return {"observations": len(values), "window": min(count, len(values)),
            "first_mean": mean(values[:count]), "last_mean": mean(values[-count:]),
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def expression(tree):
    name, children = tree["symbol"], tree.get("children", [])
    if name in {"constant", "window"}:
        return str(tree["value"])
    return name if not children else name + "(" + ",".join(expression(c) for c in children) + ")"


def generation_trace(path, initial_ids):
    records = rows(path)
    credits = {(r["tag"], r["attempt"]): r for r in records if r.get("event") == "credit"}
    groups, sources = {}, {}
    for row in records:
        if "source" not in row or "metrics" not in row:
            continue
        source = row["source"]
        group = "NEW" if source == "NEW" else "initial_formula" if source in initial_ids else "descendant_formula"
        item = groups.setdefault(group, {"attempts": 0, "credited": 0, "discoveries": 0,
                                        "duplicates": 0, "unreliable": 0, "upper_reward": 0.0, "lower_scores": []})
        credit = credits.get((row["tag"], row["attempt"]), {})
        item["attempts"] += 1
        item["credited"] += int(credit.get("credited", False))
        item["discoveries"] += int(credit.get("reward", 0) >= 1)
        item["duplicates"] += int(row["duplicate"])
        item["unreliable"] += int(not row["metrics"]["reliable"])
        item["upper_reward"] += credit.get("reward", 0)
        item["lower_scores"].append(row["metrics"]["lower_reward"])
        root = sources.setdefault(source, {"source": source, "formula": expression(row["seed_ast"]),
                                            "attempts": 0, "discoveries": 0})
        root["attempts"] += 1
        root["discoveries"] += int(credit.get("reward", 0) >= 1)
    for item in groups.values():
        item["mean_lower_reward"] = mean(item.pop("lower_scores"))
    return {"by_start_type": groups,
            "most_selected_starts": sorted(sources.values(), key=lambda x: -x["attempts"])[:10],
            "interpretation": "Observed conditional outcomes, not controlled causal effects of starting-point choice."}


def audit(output):
    target = output / "diagnostics"
    target.mkdir(exist_ok=True)
    state = read_json(output / "suite_progress.json", {})
    manifest = read_json(output / "suite_manifest.json", {})
    resources = read_json(target / "latest_resources.json", {})
    report = {"generated_at": now(), "suite": state, "resources": resources, "seeds": []}
    for root in sorted(output.glob("training_seed_*")):
        if not root.is_dir():
            continue
        config = read_json(root / "resolved_config.json", {})
        lower = [r for r in rows(root / "lower_curve.jsonl") if "iteration" in r]
        upper = [r for r in rows(root / "upper_curve.jsonl") if r.get("phase") == "upper_iteration"]
        corpus = [r for r in rows(root / "corpus.jsonl") if r.get("split") == "train"]
        initial_count = config.get("search", {}).get("initial_seeds", 0)
        initial_records = rows(Path(config["initial_library"])) if config.get("initial_library") else corpus[:initial_count]
        initial_ids = {r["id"] for r in initial_records}
        trials = []
        for path in sorted((root / "trials").glob("*.json")):
            trial = read_json(path)
            if not trial:
                continue
            future = trial.get("future", {})
            valid = [f for f in future.get("single_factors", []) if f["reliable"]]
            record = {**{k: trial.get(k) for k in ("task", "method", "new_factors", "attempts", "pool_ic",
                           "initial_pool_ic", "return", "duplicate_rate", "wall_seconds", "evaluator_calls",
                           "evaluation_seconds", "novelty_seconds", "combination_seconds",
                           "attributable_training_seconds", "new_per_end_to_end_second")},
                           "future_evaluated": "future" in trial, "future_reliable_factors": len(valid),
                           "future_mean_directed_ic": mean([f["ic"] for f in valid]),
                           "future_positive_factors": sum(f["ic"] > 0 for f in valid),
                           "future_pass_ic_threshold": sum(f["ic"] >= config["search"]["quality_threshold"] for f in valid),
                           "future_pool_ic": future.get("pool_ic"), "source": str(path.resolve())}
            if trial["method"] in {"upper_ppo", "alternating_ppo", "random", "ucb"}:
                record["generation_trace"] = generation_trace(path.with_suffix(".jsonl"), initial_ids)
            trials.append(record)
        report["seeds"].append({"name": root.name, "progress": read_json(root / "progress.json", {}),
            "config_hash": config.get("fingerprint"), "data_mode": config.get("data_mode"),
            "structure": read_json(root / "structure_validation.json"),
            "initial_formulas": [expression(r["ast"]) for r in initial_records],
            "initial_library_size": len(initial_records),
            "lower_reward": trend(lower, "mean_terminal_reward", 16),
            "lower_kl": trend(lower, "approx_kl", 16), "lower_entropy": trend(lower, "entropy", 16),
            "lower_old_probability_error": trend(lower, "old_logp_max_error", 16),
            "upper_return": trend(upper, "mean_episode_return", 4),
            "upper_discoveries_per_batch": trend(upper, "new_factors", 4),
            "upper_value_loss": trend(upper, "value_loss", 4),
            "upper_entropy": trend(upper, "entropy", 4),
            "episodes_per_batch": config.get("upper", {}).get("episodes_per_batch"),
            "trials": trials, "summary": read_json(root / "experiment_summary.json"),
            "failure": read_json(root / "failure.json")})
    complete = [s for s in report["seeds"] if s["summary"]]
    active = next((s for s in report["seeds"] if s["name"] == f"training_seed_{state.get('current_training_seed')}"), {})
    if complete and state.get("phase") != "complete":
        typical = mean([s["summary"]["end_to_end_wall_seconds"] for s in complete])
        pending = len(manifest.get("training_seeds", [])) - len(state.get("completed_training_seeds", []))
        elapsed = active.get("progress", {}).get("wall_seconds", 0)
        report["rough_remaining_hours"] = max(0, pending * typical - elapsed) / 3600
    report["recording_limits"] = [
        "早期下层每条公式的全部编辑动作、动作概率和续采样分支成绩未单独落盘，不能事后补造。",
        "检查点归档从观察器启动后开始，按轮询捕获提交版本；不是对历史所有更新的完整恢复。",
        "原始 stdout、批次曲线、上层逐候选日志及逐任务结果保持原样；诊断结果由这些文件推导。",
        "最终后续期只用于诊断与报告，观察器不回传奖励、不调参、不选择训练检查点。",
        "单个训练种子、少量任务及相关因子之间的均值差异不构成统计显著性或因果证明。"]
    atomic(target / "latest_audit.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    lines = ["# 训练持续诊断记录", "", f"更新时间：{report['generated_at']}", "",
             f"当前训练种子：{state.get('current_training_seed')}；已完成：{state.get('completed_training_seeds', [])}。",
             "观察器独立读取文件、记录资源和归档检查点，不改变训练过程。", ""]
    if resources:
        lines += ["## 资源", "", "```json", json.dumps(resources, ensure_ascii=False, indent=2), "```", ""]
    if "rough_remaining_hours" in report:
        lines += [f"按已完成种子的整套实测耗时粗估，剩余约 {report['rough_remaining_hours']:.1f} 小时；随负载和采样变化。", ""]
    for seed in report["seeds"]:
        lines += [f"## {seed['name']}", "", "### 当前阶段与训练趋势", "", "```json",
                  json.dumps({k: seed[k] for k in ("progress", "structure", "lower_reward", "lower_kl", "lower_entropy",
                     "lower_old_probability_error", "upper_return", "upper_discoveries_per_batch", "upper_value_loss", "upper_entropy")},
                     ensure_ascii=False, indent=2), "```", "", "### 初始公式", ""]
        lines += [f"{i}. `{text}`" for i, text in enumerate(seed["initial_formulas"], 1)]
        lines += ["", "### 搜索与锁定后续期结果", "",
                  "| 任务 | 方法 | 搜索期发现数 | 搜索组合IC | 后续可靠数 | 后续达IC门槛数 | 后续平均定向IC | 后续组合IC |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        def fmt(value):
            return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        for trial in seed["trials"]:
            keys = ("task", "method", "new_factors", "pool_ic", "future_reliable_factors", "future_pass_ic_threshold",
                    "future_mean_directed_ic", "future_pool_ic")
            lines.append("| " + " | ".join(fmt(trial[k]) if trial["future_evaluated"] or not k.startswith("future_") else "—"
                                         for k in keys) + " |")
        lines += ["", "数据模式：" + str(seed["data_mode"]) + "。完整成本和原始路径见同目录 latest_audit.json。", ""]
    lines += ["## 判断问题时使用的证据", "",
        "1. 下层奖励是否上升，同时上层发现数或回报下降：这是目标不一致的线索，需区分种子噪声与任务差异。",
        "2. 固定下层时，上层是否稳定优于随机、质量贪心和UCB：用于检验起点调度的额外价值。",
        "3. 搜索期达标数是否在锁定后续期保留：用于区分历史拟合与跨时期有效性。",
        "4. 交替训练是否优于固定下层：用于检验协同更新的实际收益。",
        "5. 搜索速度和计入训练成本的端到端速度分别比较，资源利用率不直接等于算法效率。", "",
        "## 记录边界", "", *[f"- {note}" for note in report["recording_limits"]], ""]
    atomic(target / "latest_audit.md", "\n".join(lines))
    append(target / "audit_history.jsonl", {"time": report["generated_at"], "suite": state,
           "seeds": [{"name": s["name"], "progress": s["progress"], "trial_count": len(s["trials"])} for s in report["seeds"]]})
    return report


def resources(pid):
    command = f"""$taskP = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue
$taskO = Get-CimInstance Win32_OperatingSystem
$taskM = Get-CimInstance Win32_ComputerSystem
$taskC = Get-CimInstance Win32_Processor
$taskList = Get-CimInstance Win32_Process
$taskIds = @({int(pid)})
do {{
  $taskCount = $taskIds.Count
  $taskIds = @(@($taskIds) + @($taskList | Where-Object {{ $_.ParentProcessId -in $taskIds }} | Select-Object -ExpandProperty ProcessId) | Sort-Object -Unique)
}} while ($taskIds.Count -gt $taskCount)
$taskGroup = @(Get-Process -Id $taskIds -ErrorAction SilentlyContinue)
$taskSystemCpu = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object Name -eq '_Total'
[pscustomobject]@{{time=(Get-Date).ToString('o');exists=($null -ne $taskP);pid={int(pid)};cpu=($taskC.Name -join ', ');logical_processors=$taskM.NumberOfLogicalProcessors;cpu_seconds=$taskP.CPU;working_set_gb=[math]::Round($taskP.WorkingSet64/1GB,3);private_gb=[math]::Round($taskP.PrivateMemorySize64/1GB,3);memory_total_gb=[math]::Round($taskM.TotalPhysicalMemory/1GB,2);memory_free_gb=[math]::Round($taskO.FreePhysicalMemory/1MB,2);group_pids=@($taskIds);group_processes=$taskGroup.Count;group_cpu_seconds=($taskGroup | Measure-Object CPU -Sum).Sum;group_private_gb=[math]::Round(($taskGroup | Measure-Object PrivateMemorySize64 -Sum).Sum/1GB,3);group_working_set_sum_gb=[math]::Round(($taskGroup | Measure-Object WorkingSet64 -Sum).Sum/1GB,3);system_cpu_percent=$taskSystemCpu.PercentProcessorTime;process_start_utc=if($taskP){{$taskP.StartTime.ToUniversalTime().ToString('o')}}else{{$null}}}} | ConvertTo-Json -Compress"""
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW,
                               timeout=30, check=True)
    return json.loads(completed.stdout.decode("utf-8-sig", errors="replace"))


def archive(output, seen):
    target = output / "diagnostics"
    for source in sorted(output.glob("training_seed_*/**/*.pt")):
        before = source.stat()
        key = str(source.relative_to(output))
        signature = (before.st_size, before.st_mtime_ns)
        if seen.get(key) == signature:
            continue
        destination = target / "checkpoints" / source.relative_to(output).parent / f"{source.stem}__{before.st_mtime_ns}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        try:
            shutil.copyfile(source, temporary)
            after = source.stat()
            if signature != (after.st_size, after.st_mtime_ns):
                continue
            os.replace(temporary, destination)
            append(target / "checkpoint_manifest.jsonl", {"time": now(), "source": key,
                "source_signature": signature, "snapshot": str(destination.relative_to(output)),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bytes": before.st_size})
            seen[key] = signature
        finally:
            temporary.unlink(missing_ok=True)


def watch(output, pid, interval):
    import msvcrt
    target = output / "diagnostics"
    target.mkdir(exist_ok=True)
    lock = (target / ".observer.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    launch = read_json(output / "launch.json", {})
    if launch.get("pid") != pid:
        raise ValueError("observer PID differs from the recorded training process")
    append(target / "observer_events.jsonl", {"time": now(), "event": "started", "observer_pid": os.getpid(), "training_pid": pid})
    seen = {r["source"]: tuple(r["source_signature"]) for r in rows(target / "checkpoint_manifest.jsonl")}
    previous_resource, resource_at, report_at, trial_count = None, 0.0, 0.0, -1
    while True:
        try:
            archive(output, seen)
            current = time.monotonic()
            alive = True
            if current - resource_at >= 60:
                reading = resources(pid)
                alive = reading["exists"]
                if alive:
                    started = datetime.fromisoformat(reading["process_start_utc"])
                    if abs((started - datetime.fromisoformat(launch["started_at"])).total_seconds()) > 5:
                        raise RuntimeError("training PID was reused by another process")
                if previous_resource and alive:
                    seconds = current - resource_at
                    core_percent = 100 * (reading["cpu_seconds"] - previous_resource["cpu_seconds"]) / seconds
                    reading["process_cpu_percent_one_core_100"] = round(core_percent, 2)
                    reading["process_cpu_percent_machine"] = round(core_percent / reading["logical_processors"], 2)
                    if reading["group_pids"] == previous_resource.get("group_pids"):
                        reading["group_cpu_percent_machine"] = round(100 * (reading["group_cpu_seconds"] -
                            previous_resource["group_cpu_seconds"]) / seconds / reading["logical_processors"], 2)
                append(target / "resources.jsonl", reading)
                atomic(target / "latest_resources.json", json.dumps(reading, ensure_ascii=False, indent=2))
                previous_resource, resource_at = reading, current
            state = read_json(output / "suite_progress.json", {})
            count = sum(1 for _ in output.glob("training_seed_*/trials/*.json"))
            finished = state.get("phase") == "complete" or not alive
            if current - report_at >= 300 or count != trial_count or finished:
                report = audit(output)
                print(json.dumps({"time": now(), "event": "audit_saved", "trial_count": count,
                                  "current_seed": report["suite"].get("current_training_seed")}), flush=True)
                report_at, trial_count = current, count
            if finished:
                append(target / "observer_events.jsonl", {"time": now(), "event": "finished", "suite_complete": state.get("phase") == "complete"})
                break
        except Exception as error:
            append(target / "observer_events.jsonl", {"time": now(), "event": "error", "type": type(error).__name__, "message": str(error)})
            print(f"Observer error: {type(error).__name__}: {error}", flush=True)
            if isinstance(error, RuntimeError):
                raise
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "watch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "launch.json").is_file():
        parser.error("output must be an existing launched training suite")
    if args.command == "watch":
        if args.pid is None or args.interval < 1:
            parser.error("watch requires --pid and an interval of at least one second")
        watch(output, args.pid, args.interval)
    else:
        audit(output)
        print(output / "diagnostics" / "latest_audit.md")
