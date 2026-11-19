"""从公司数据库快照准备行情，并按标签成熟时间生成滚动实验任务。"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import exchange_calendars as calendars
import numpy as np
import pandas as pd

from .io import digest, save_json
from .selection_config import file_hash


FIELDS = ["open", "close", "high", "low", "volume", "amount", "TurnoverRate"]
TRAINING_START = "2025-07-01"


def latest_completed_session(now=None):
    """按北京时间收盘时点返回最近已结束的交易日，最新数据限于2026年。"""
    stamp = pd.Timestamp.now(tz="Asia/Shanghai") if now is None else pd.Timestamp(now)
    stamp = stamp.tz_localize("Asia/Shanghai") if stamp.tzinfo is None else stamp.tz_convert("Asia/Shanghai")
    end = stamp.tz_localize(None).normalize()
    if stamp.hour < 15:
        end -= pd.Timedelta(days=1)
    end = min(end, pd.Timestamp("2026-12-31"))
    calendar = calendars.get_calendar("XSHG", start=str((end - pd.Timedelta(days=30)).date()), end=str(end.date()))
    return str(calendar.sessions[-1].date())


def design_tasks(cache, *, start=None, development_cutoff=None,
                 test_cutoff=None, horizon=20, block=20, stride=10):
    """默认用全部成熟历史任务训练并比较，显式开发截止日沿用历史保留划分。"""
    if min(horizon, block, stride) < 1:
        raise ValueError("positive horizon, block and stride are required")
    manifest_path = Path(cache) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    cutoff_source = "calendar_plan"
    if manifest.get("date_end"):
        cutoff_source = "cache_manifest"
    if test_cutoff:
        cutoff_source = "explicit"
    start = start or manifest.get("experiment_training_start", TRAINING_START)
    test_cutoff = test_cutoff or manifest.get("date_end") or latest_completed_session()
    if pd.Timestamp(start) > pd.Timestamp(test_cutoff):
        raise ValueError("training start exceeds available data cutoff")
    calendar = calendars.get_calendar("XSHG", start=f"{pd.Timestamp(start).year}-01-01", end=test_cutoff)
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
                   "information_cutoff_source": cutoff_source,
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
    """为已有历史实验保留主缓存导入，标签计算与数据库入口共用。"""
    import pyarrow.parquet as pq
    master = Path(master).resolve()
    codes = read_codes(codes_file)
    frames, hashes = [], {}
    for code in codes:
        path = master / "stocks" / f"{code}.parquet"
        available = set(pq.ParquetFile(path).schema.names)
        required = {"date", "ycp", *FIELDS}
        if not required <= available:
            raise ValueError(f"{code}: missing {sorted(required - available)}")
        optional = available & {"Ifsuspend", "if_up", "if_down"}
        frame = pd.read_parquet(path, columns=sorted(required | optional))
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[(frame.date >= start) & (frame.date <= end)]
        if frame.date.duplicated().any():
            raise ValueError(f"duplicate stock/date rows: {code}")
        frames.append(frame.assign(code=str(code)))
        hashes[str(code)] = file_hash(path)
    frame = pd.concat(frames, ignore_index=True)
    source_meta = master / "meta.json"
    conventions = json.loads(source_meta.read_text(encoding="utf-8")).get("data_conventions_verified", False) if source_meta.exists() else False
    source = {"data_source": "historical_master", "source_fingerprint": digest(hashes),
              "data_conventions_verified": conventions}
    return write_panel(frame, codes, output, start, end, horizon, source, hashes)


def read_codes(path):
    """读取固定股票池，拒绝空池、重复及无效证券代码。"""
    return checked_codes(json.loads(Path(path).read_text(encoding="utf-8")))


def checked_codes(values):
    """所有行情来源统一使用互不重复的六位股票代码。"""
    from repositories.local_store import normalize_code
    codes = [normalize_code(code) for code in values]
    if not codes or len(set(codes)) != len(codes) or any(len(code) != 6 or not code.isdigit() for code in codes):
        raise ValueError("codes must be distinct six-digit stock identifiers")
    return codes


def warmup_start(training_start, lookback):
    """只为首个任务额外读取足够的历史交易日，供时序算子预热。"""
    start = pd.Timestamp(training_start)
    calendar = calendars.get_calendar("XSHG", start=str((start - pd.Timedelta(days=2 * lookback + 90)).date()),
                                     end=str((start + pd.Timedelta(days=10)).date()))
    dates = calendar.sessions.tz_localize(None)
    first = int(dates.searchsorted(start))
    return str(dates[first - lookback].date())


def prepare_database_panel(codes_file, output, *, training_start=TRAINING_START, start=None, end=None,
                           horizon=20, lookback=100, snapshot_id=None, snapshot_root=None, refresh=False):
    """复用上游固定快照，冻结最新成熟行情及数据来源后供所有算法共用。"""
    from repositories import clickhouse, snapshot_store
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError("use an empty output directory for immutable panel preparation")
    codes = read_codes(codes_file) if codes_file is not None else None
    universe = "explicit_codes" if codes is not None else "all_database_stocks"
    if min(horizon, lookback) < 1:
        raise ValueError("positive horizon and lookback are required")
    required_start = warmup_start(training_start, lookback)
    start = str(pd.Timestamp(start or required_start).date())
    if start > required_start:
        raise ValueError(f"warmup requires data from {required_start} or earlier")
    fields = [*FIELDS, "ycp", "Ifsuspend", "if_up", "if_down"]
    if snapshot_id:
        if refresh:
            raise ValueError("choose an existing snapshot or refresh from the database")
        snapshot = snapshot_store.snapshot_manifest(snapshot_id, snapshot_root=snapshot_root)
        end = str(pd.Timestamp(end or snapshot["date_max"]).date())
        if codes is None:
            codes, universe = checked_codes(snapshot["codes"]), "snapshot_codes"
        elif set(codes) != set(snapshot["codes"]):
            raise ValueError("snapshot stock universe differs from experiment codes")
        if start < snapshot["date_min"] or end > snapshot["date_max"]:
            raise ValueError("snapshot does not cover the requested experiment interval")
    else:
        ceiling = latest_completed_session()
        requested_end = str(pd.Timestamp(end or ceiling).date())
        if requested_end > ceiling:
            raise ValueError(f"latest completed trading day is {ceiling}")
        settings = clickhouse.resolve_settings()
        before = str((pd.Timestamp(requested_end) + pd.Timedelta(days=1)).date())
        client = clickhouse.ClickHouseHTTPClient(settings)
        rows = client.query(
            f"SELECT toString(max(toDate(timestamps))) AS date_max FROM {clickhouse.DAILY_TABLE} "
            f"WHERE category = 'stock' AND timestamps < '{before}'")
        available = rows[0].get("date_max") if rows else None
        if not available or pd.Timestamp(available) < pd.Timestamp(training_start):
            raise ValueError("database has no recent data covering the experiment start")
        if end and str(pd.Timestamp(available).date()) < requested_end:
            raise ValueError(f"database data ends at {available}, before requested {requested_end}")
        end = str(pd.Timestamp(available).date())
        if codes is None:
            before = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
            stocks = client.query(
                f"SELECT DISTINCT toString(code) AS code FROM {clickhouse.DAILY_TABLE} "
                f"WHERE category = 'stock' AND timestamps >= '{start}' AND timestamps < '{before}' ORDER BY code")
            if not stocks:
                raise ValueError("database has no stocks in the requested panel period")
            codes = sorted(checked_codes(row["code"] for row in stocks))
        snapshot = snapshot_store.create_or_reuse_snapshot(codes, fields, start, end,
            pool_id=digest(codes), snapshot_root=snapshot_root, as_of=end, refresh=refresh, category="stock")
    if end > latest_completed_session():
        raise ValueError("snapshot includes an unfinished or future trading day")
    frame = snapshot_store.load_snapshot(snapshot["snapshot_id"], columns=["date", "code", *fields],
                                         snapshot_root=snapshot_root)
    frame = frame[(frame.date >= pd.Timestamp(start)) & (frame.date <= pd.Timestamp(end))]
    source = {"data_source": "clickhouse_snapshot", "stock_universe": universe,
              "source_snapshot_id": snapshot["snapshot_id"],
              "source_fingerprint": file_hash(snapshot_store.snapshot_path(snapshot["snapshot_id"], snapshot_root=snapshot_root)),
              "source_snapshot": snapshot, "experiment_training_start": training_start,
              "data_conventions_verified": False}
    return write_panel(frame, codes, output, start, end, horizon, source)


def write_panel(frame, codes, output, start, end, horizon, source, hashes=None):
    """统一校验行情、构造因果收益标签，完整写入后原子发布训练面板。"""
    from ..prepare_budget_study import reference_panel, forward_labels, missing_turnover
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use an empty output directory for immutable panel preparation")
    required = {"date", "code", "ycp", *FIELDS}
    if not required <= set(frame.columns):
        raise ValueError(f"source fields missing: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.empty or frame.duplicated(["date", "code"]).any():
        raise ValueError("source is empty or has duplicate stock/date rows")
    missing_codes = set(codes) - set(frame["code"].astype(str))
    if missing_codes:
        raise ValueError(f"source is missing requested stocks: {sorted(missing_codes)[:8]}")
    dates = calendars.get_calendar("XSHG", start=start, end=end).sessions.tz_localize(None)
    missing = dates.difference(pd.DatetimeIndex(frame.date.unique()))
    if len(missing):
        raise ValueError(f"source has missing whole-market sessions, including {missing[:8].strftime('%Y-%m-%d').tolist()}; obtain complete source data")
    flags = set(frame.columns) & {"Ifsuspend", "if_up", "if_down"}
    arrays = {field: frame.pivot(index="date", columns="code", values=field).reindex(index=dates, columns=codes).to_numpy(float)
              for field in [*FIELDS, "ycp", *sorted(flags)]}
    arrays["TurnoverRate"], unavailable = missing_turnover(arrays["TurnoverRate"], arrays["volume"])
    adjusted, returns, _ = reference_panel({name: arrays[name] for name in FIELDS}, arrays["ycp"])
    labels = forward_labels(returns, horizon)
    labels[~np.isfinite(arrays["close"]) | (arrays["close"] <= 0)] = np.nan
    manifest = {"fields": [*FIELDS, *sorted(flags)], "n_stocks": len(codes), "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()), "calendar": "XSHG", "label_file": f"label_h{horizon}.npy",
        "label_horizon": horizon, "label_mode": "reference_return_chain", **source,
        "available_execution_flags": sorted(flags),
        "unavailable_turnover_dates": dates[unavailable].strftime("%Y-%m-%d").tolist()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}.building-", dir=output.parent) as temporary:
        panel = Path(temporary) / "panel"
        panel.mkdir()
        for name, values in {**adjusted, **{name: arrays[name] for name in flags}}.items():
            np.save(panel / f"{name}.npy", values)
        np.save(panel / "dates.npy", dates.values)
        np.save(panel / manifest["label_file"], labels)
        save_json(panel / "codes.json", codes)
        save_json(panel / "manifest.json", manifest)
        if hashes is not None:
            save_json(panel / "source_hashes.json", hashes)
        if output.exists():
            output.rmdir()
        panel.rename(output)
    return manifest
