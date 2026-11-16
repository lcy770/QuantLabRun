"""Freeze calendar-aligned, causal reference-return data and a rolling protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import exchange_calendars as calendars
import numpy as np
import pandas as pd

from .hierarchical.io import save_json


def reference_panel(raw, references):
    """Causal forward adjustment; no future prices determine past factors."""
    close = np.asarray(raw["close"], dtype=float)
    ref = np.asarray(references, dtype=float)
    observed = np.isfinite(close) & (close > 0) & np.isfinite(ref) & (ref > 0)
    factors = np.ones_like(close)
    for t in range(1, len(close)):
        valid = observed[t] & np.isfinite(close[t - 1]) & (close[t - 1] > 0)
        factors[t] = factors[t - 1] * np.where(valid, close[t - 1] / np.where(valid, ref[t], 1), 1)
    result = {name: np.asarray(values, dtype=float).copy() for name, values in raw.items()}
    for name in ("open", "close", "high", "low"):
        result[name] *= factors
    result["volume"] /= factors
    returns = np.where(observed, close / np.where(observed, ref, 1) - 1, np.nan)
    return result, returns, factors


def forward_labels(returns, horizon):
    """Require every official session in (t,t+H], never bridge missing data."""
    result = np.full_like(returns, np.nan, dtype=np.float64)
    if len(returns) > horizon:
        windows = np.lib.stride_tricks.sliding_window_view(returns[1:], horizon, axis=0)
        valid = np.isfinite(windows).all(-1) & (windows > -1).all(-1)
        values = np.prod(1 + np.where(np.isfinite(windows), windows, 0), axis=-1) - 1
        result[:len(values)] = np.where(valid, values, np.nan)
    return result


def missing_turnover(values, volume):
    """Whole-market zero turnover with positive trading volume is unavailable data."""
    values = np.asarray(values, dtype=float).copy()
    active = np.isfinite(volume) & (np.asarray(volume) > 0)
    missing = active.any(axis=1) & np.all(~active | (values == 0), axis=1)
    values[missing] = np.nan
    return values, missing


def prepare(master, codes_path, output):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("use a new data directory; frozen panels are immutable")
    output.mkdir(parents=True, exist_ok=True)
    codes = json.loads(Path(codes_path).read_text(encoding="utf-8"))
    fields = ["open", "close", "high", "low", "volume", "amount", "TurnoverRate"]
    frames, sources = [], {}
    for code in codes:
        path = Path(master) / "stocks" / f"{code}.parquet"
        sources[code] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path, columns=["date", *fields, "ycp", "ChangePCT"])
        if frame.date.duplicated().any():
            raise ValueError(f"duplicate dates: {code}")
        frames.append(frame.assign(code=code))
    frame = pd.concat(frames, ignore_index=True)
    sessions = calendars.get_calendar("XSHG", start="2021-06-18", end="2026-06-16").sessions.tz_localize(None)
    source_days = pd.DatetimeIndex(frame.date.unique()).sort_values()
    missing = sessions.difference(source_days)
    extra = source_days.difference(sessions)
    # The study is capped before the globally missing 2026 sessions.
    dates = sessions[sessions <= "2026-02-13"]
    raw = {name: frame.pivot(index="date", columns="code", values=name).reindex(index=dates, columns=codes).to_numpy(float)
           for name in [*fields, "ycp", "ChangePCT"]}
    raw["TurnoverRate"], missing_turnover_days = missing_turnover(raw["TurnoverRate"], raw["volume"])
    close, ref = raw["close"], raw["ycp"]
    previous = np.vstack((np.full((1, len(codes)), np.nan), close[:-1]))
    error = abs(100 * (close / ref - 1) - raw["ChangePCT"])
    valid = np.isfinite(error)
    if np.nanquantile(error, .99) > .02 or len(extra):
        raise ValueError("calendar/reference return consistency failed")
    adjusted, daily, factors = reference_panel({k: raw[k] for k in fields}, ref)
    labels = forward_labels(daily, 20)
    labels[~np.isfinite(close) | (close <= 0)] = np.nan
    for name, values in adjusted.items():
        np.save(output / f"{name}.npy", values.astype(np.float32))
    np.save(output / "label_h20.npy", labels)
    np.save(output / "reference_daily_return.npy", daily)
    np.save(output / "adjustment_factor.npy", factors)
    np.save(output / "dates.npy", dates.to_numpy(dtype="datetime64[ns]"))
    save_json(output / "codes.json", codes)
    audit = {"source_hashes": sources, "codes_source": str(Path(codes_path).resolve()),
             "unavailable_turnover_dates": [str(day.date()) for day in dates[missing_turnover_days]],
             "rows": len(frame), "stocks": len(codes), "calendar": "XSHG",
             "calendar_package_version": calendars.__version__, "calendar_missing_sessions": [str(d.date()) for d in missing],
             "calendar_extra_sessions": [str(d.date()) for d in extra],
             "reference_return_error_p99_percentage_points": float(np.quantile(error[valid], .99)),
             "reference_diff_gt_half_percent": int((abs(previous / ref - 1) > .005).sum()),
             "source_date_range": [str(source_days[0].date()), str(source_days[-1].date())],
             "study_date_range": [str(dates[0].date()), str(dates[-1].date())],
             "adjustment": "causal previous-close/reference-close chain; volume inverse-scaled; amount unchanged",
             "target": "product of next 20 official-session close/reference returns; all 20 required",
             "unresolved": ["delisted/security-history completeness", "reference-close cash/reinvestment convention"],
             "previous_future_periods_already_observed": True}
    old_target = np.full_like(close, np.nan)
    old_target[:-20] = close[20:] / close[:-20] - 1
    difference = abs(labels - old_target)
    audit["old_new_label_abs_diff_p99"] = float(np.nanquantile(difference, .99))
    audit["old_new_label_diff_over_001_fraction"] = float(np.nanmean(np.where(np.isfinite(difference), difference > .01, np.nan)))
    save_json(output / "data_audit.json", audit)
    save_json(output / "manifest.json", {"fields": fields, "n_stocks": len(codes),
        "label_file": "label_h20.npy", "label_horizon": 20, "label_mode": "reference_return_chain",
        "date_start": str(dates[0].date()), "date_end": str(dates[-1].date()),
        "calendar": "XSHG", "data_conventions_verified": False,
        "source_fingerprint": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()})
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.master, args.codes, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, ensure_ascii=False), flush=True)
