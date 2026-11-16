"""按实际交易日与标签成熟生成截至2026年6月16日的滚动实验任务及数据缓存。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as calendars
import numpy as np
import pandas as pd

from .io import digest, save_json
from .selection_config import file_hash


FIELDS = ["open", "close", "high", "low", "volume", "amount", "TurnoverRate"]


def design_tasks(cache, *, start="2024-01-02", development_cutoff=None,
                 test_cutoff="2026-06-16", horizon=20, block=20, stride=10):
    """默认用全部成熟历史任务训练并比较，显式开发截止日沿用历史保留划分。"""
    if min(horizon, block, stride) < 1:
        raise ValueError("positive horizon, block and stride are required")
    calendar = calendars.get_calendar("XSHG", start="2024-01-01", end=test_cutoff)
    dates = calendar.sessions.tz_localize(None).values.astype("datetime64[D]")
    start_index = int(np.searchsorted(dates, np.datetime64(start)))
    test_end = int(np.searchsorted(dates, np.datetime64(test_cutoff), side="right")) - 1

    def section(first, last):
        """返回可直接加载的信号日期范围。"""
        if first < 0 or last < first or last + horizon >= len(dates):
            raise ValueError("insufficient calendar for mature scoring sections")
        return {"cache": str(cache), "start": str(dates[first]), "end": str(dates[last])}

    if development_cutoff is None:
        span = 4 * block + 4 * horizon
        last_first = test_end - span + 1
        starts = list(range(start_index, last_first + 1, stride))
        if not starts:
            raise ValueError("no complete rolling F/G/A/R task fits the training period")
        if starts[-1] != last_first:
            starts.append(last_first)
        tasks = []
        for first in starts:
            task = {"name": f"train_{len(tasks):03d}", "role": "train"}
            for index, name in enumerate(("fit", "search", "validation", "feedback")):
                begin = first + index * (block + horizon)
                task[name] = section(begin, begin + block - 1)
            tasks.append(task)
        summary = {"mode": "shared_training_comparison", "training_information_start": str(dates[start_index]),
                   "training_label_cutoff": str(dates[test_end]), "comparison_information_cutoff": str(dates[test_end]),
                   "latest_signal_date": str(dates[test_end - horizon]), "training_tasks": len(tasks),
                   "comparison_tasks": len(tasks), "horizon": horizon, "block_days": block,
                   "rolling_stride": stride, "comparison_uses_training_data": True,
                   "independent_test_available": False}
        return tasks, summary

    development_end = int(np.searchsorted(dates, np.datetime64(development_cutoff), side="right")) - 1

    def backwards(last, names):
        """前一段结束标签严格早于后一段开始信号，保留完整 20 日成熟等待。"""
        result = {}
        for name in reversed(names):
            first = last - block + 1
            result[name] = section(first, last)
            last = first - horizon - 1
        return result

    development = {"name": "development", "role": "development",
                   **backwards(development_end - horizon, ("fit", "search", "validation", "feedback"))}
    train_cutoff = int(np.searchsorted(dates, np.datetime64(development["search"]["start"]))) - 1
    tasks = []
    span = 4 * block + 4 * horizon
    for first in range(start_index, train_cutoff - span + 2, stride):
        task = {"name": f"train_{len(tasks):03d}", "role": "train"}
        for index, name in enumerate(("fit", "search", "validation", "feedback")):
            begin = first + index * (block + horizon)
            task[name] = section(begin, begin + block - 1)
        tasks.append(task)
    if not tasks:
        raise ValueError("no complete rolling F/G/A/R task fits the training period")
    test = {"name": "locked_test", "role": "test",
            **backwards(development_end - horizon, ("fit", "search", "validation")),
            "future": section(development_end + 1, test_end - horizon)}
    summary = {"training_information_start": str(dates[start_index]),
               "training_label_cutoff": str(dates[train_cutoff]),
               "development_information_cutoff": str(dates[development_end]),
               "test_information_cutoff": str(dates[test_end]),
               "test_signal_start": test["future"]["start"], "test_signal_end": test["future"]["end"],
               "training_tasks": len(tasks), "horizon": horizon, "block_days": block,
               "rolling_stride": stride, "development_labels_used_for_training": False}
    return [*tasks, development, test], summary


def validate_cache(cache, tasks, lookback, horizon, fields):
    """验证真实缓存覆盖范围、交易日、股票身份及侧载标签形状，不补齐缺失行情。"""
    cache = Path(cache).resolve()
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    dates = np.load(cache / "dates.npy").astype("datetime64[D]")
    if len(dates) < lookback + horizon + 3 or (np.diff(dates).astype(int) <= 0).any():
        raise ValueError("cache dates must be strictly increasing with sufficient history")
    codes = json.loads((cache / "codes.json").read_text(encoding="utf-8"))
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("stock codes must be nonempty and unique")
    earliest = min(np.datetime64(task[name]["start"]) for task in tasks
                   for name in ("fit", "search", "validation", "feedback", "future") if name in task)
    latest = max(np.datetime64(task[name]["end"]) for task in tasks
                 for name in ("fit", "search", "validation", "feedback", "future") if name in task)
    if dates[lookback] > earliest or dates[-horizon - 1] < latest:
        raise ValueError(f"data coverage insufficient: scoring {dates[lookback]}..{dates[-horizon - 1]}, required {earliest}..{latest}; update the cache before formal experiments")
    expected = calendars.get_calendar("XSHG", start=str(dates[0]), end=str(dates[-1])).sessions.tz_localize(None).values.astype("datetime64[D]")
    if not np.array_equal(dates, expected):
        raise ValueError("cache must contain every official XSHG session without inserted dates")
    shape = (len(dates), len(codes))
    for field in fields:
        if field not in manifest["fields"] or np.load(cache / f"{field}.npy", mmap_mode="r").shape != shape:
            raise ValueError(f"missing/misaligned predictor {field}")
    if manifest.get("label_file"):
        if manifest.get("label_horizon") != horizon or np.load(cache / manifest["label_file"], mmap_mode="r").shape != shape:
            raise ValueError("sidecar label horizon/shape differs from protocol")
    return {"cache": str(cache), "rows": len(dates), "stocks": len(codes),
            "start": str(dates[0]), "end": str(dates[-1]),
            "data_conventions_verified": manifest.get("data_conventions_verified", False)}


def prepare_panel(master, codes_file, output, *, start="2023-07-03", end="2026-06-16", horizon=20):
    """复用参考价收益链构建完整交易日缓存，并保存实际可用交易标记。"""
    from ..prepare_budget_study import reference_panel, forward_labels, missing_turnover
    import pyarrow.parquet as pq
    output, master = Path(output).resolve(), Path(master).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use an empty output directory for immutable panel preparation")
    codes = json.loads(Path(codes_file).read_text(encoding="utf-8"))
    if len(set(codes)) != len(codes):
        raise ValueError("duplicate stock codes")
    dates = calendars.get_calendar("XSHG", start=start, end=end).sessions.tz_localize(None)
    frames, hashes, flags = [], {}, set()
    for code in codes:
        path = master / "stocks" / f"{code}.parquet"
        available = set(pq.ParquetFile(path).schema.names)
        required = {"date", "ycp", *FIELDS}
        if not required <= available:
            raise ValueError(f"{code}: missing {sorted(required - available)}")
        optional = available & {"Ifsuspend", "if_up", "if_down"}
        flags.update(optional)
        frame = pd.read_parquet(path, columns=sorted(required | optional))
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame.date >= dates[0]) & (frame.date <= dates[-1])]
        if frame.date.duplicated().any():
            raise ValueError(f"duplicate stock/date rows: {code}")
        frames.append(frame.assign(code=str(code)))
        hashes[str(code)] = file_hash(path)
    frame = pd.concat(frames, ignore_index=True)
    missing = dates.difference(pd.DatetimeIndex(frame.date.unique()))
    if len(missing):
        raise ValueError(f"source has missing whole-market sessions, including {missing[:8].strftime('%Y-%m-%d').tolist()}; obtain complete source data")
    arrays = {field: frame.pivot(index="date", columns="code", values=field).reindex(index=dates, columns=codes).to_numpy(float)
              for field in [*FIELDS, "ycp", *sorted(flags)]}
    arrays["TurnoverRate"], unavailable = missing_turnover(arrays["TurnoverRate"], arrays["volume"])
    adjusted, returns, _ = reference_panel({name: arrays[name] for name in FIELDS}, arrays["ycp"])
    labels = forward_labels(returns, horizon)
    labels[~np.isfinite(arrays["close"]) | (arrays["close"] <= 0)] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    for name, values in {**adjusted, **{name: arrays[name] for name in flags}}.items():
        np.save(output / f"{name}.npy", values)
    np.save(output / "dates.npy", dates.values)
    np.save(output / "label_h20.npy", labels)
    save_json(output / "codes.json", codes)
    source_meta = master / "meta.json"
    conventions = json.loads(source_meta.read_text(encoding="utf-8")).get("data_conventions_verified", False) if source_meta.exists() else False
    manifest = {"fields": [*FIELDS, *sorted(flags)], "n_stocks": len(codes), "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()), "calendar": "XSHG", "label_file": "label_h20.npy",
        "label_horizon": horizon, "label_mode": "reference_return_chain", "source_fingerprint": digest(hashes),
        "data_conventions_verified": conventions, "available_execution_flags": sorted(flags),
        "unavailable_turnover_dates": dates[unavailable].strftime("%Y-%m-%d").tolist()}
    save_json(output / "manifest.json", manifest)
    save_json(output / "source_hashes.json", hashes)
    return manifest
