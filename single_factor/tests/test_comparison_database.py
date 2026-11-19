"""核验数据库快照入口、最新日期划分及20日标签，行情为合成数据。"""
import json

import exchange_calendars as calendars
import numpy as np
import pandas as pd
import pytest

from repositories import clickhouse, company_source, snapshot_store
from single_factor.hierarchical import comparison_cli, comparison_data
from single_factor.hierarchical.io import save_json


def test_rolling_tasks_follow_snapshot_dates(tmp_path):
    """新默认窗口读取面板截止日，历史协议仍能显式指定原日期。"""
    save_json(tmp_path / "manifest.json", {"experiment_training_start": "2025-07-01", "date_end": "2026-11-16"})
    tasks, summary = comparison_data.design_tasks(tmp_path)
    assert summary["information_cutoff_source"] == "cache_manifest"
    assert summary["training_information_start"] == "2025-07-01"
    assert summary["training_label_cutoff"] == "2026-11-16"
    assert summary["latest_signal_date"] == "2026-10-19"
    assert len(tasks) == 19
    assert all(task["role"] == "train" for task in tasks)
    dates = calendars.get_calendar("XSHG", start="2025-01-01", end="2026-11-16").sessions.tz_localize(None)
    for task in tasks:
        sections = [task[name] for name in ("fit", "search", "validation", "feedback")]
        for previous, current in zip(sections, sections[1:]):
            assert dates[dates.get_loc(previous["end"]) + 20] < pd.Timestamp(current["start"])
    assert dates[dates.get_loc(tasks[-1]["feedback"]["end"]) + 20] == pd.Timestamp("2026-11-16")


def test_latest_session_waits_for_close():
    """盘中、周末及跨年运行时，避免把未结束或2027年的行情纳入本轮。"""
    assert comparison_data.latest_completed_session("2026-11-17 14:59") == "2026-11-16"
    assert comparison_data.latest_completed_session("2026-11-17 15:01") == "2026-11-17"
    assert comparison_data.latest_completed_session("2026-11-14 17:00") == "2026-11-13"
    assert comparison_data.latest_completed_session("2027-01-04 17:00") == "2026-12-31"


@pytest.mark.parametrize("missing_day,all_stocks", [(False, False), (True, False), (False, True)])
def test_database_cli_and_offline_snapshot(tmp_path, monkeypatch, missing_day, all_stocks):
    """通过真实CLI、上游快照及面板发布链路，验证标签和缺日失败行为。"""
    import sys
    codes = [f"{i:06d}" for i in range(1, 502)] if all_stocks else ["000001", "000002", "000004", "000005", "000006"]
    codes_path = tmp_path / "codes.json"
    save_json(codes_path, codes)
    queries, captures = [], []

    def query(client, sql):
        """模拟最新日期和不设名单上限的全市场代码查询。"""
        queries.append(sql)
        if "SELECT DISTINCT" in sql:
            assert all_stocks and "LIMIT" not in sql
            assert "category = 'stock'" in sql and "timestamps >= '2025-01-27'" in sql
            assert "timestamps < '2026-11-17'" in sql
            return [{"code": code} for code in reversed(codes)]
        return [{"date_max": "2026-11-16"}]

    def capture(stock_codes, fields, output, start, end, *, category):
        """输出覆盖交易日的合成源快照，缺日分支用于验证拒绝不完整数据。"""
        dates = calendars.get_calendar("XSHG", start=start, end=end).sessions.tz_localize(None)
        if missing_day:
            dates = dates.delete(110)
        captures.append((start, end))
        assert category == "stock" and list(stock_codes) == codes
        panel_sql = company_source.build_panel_sql(stock_codes, fields, start, end, category=category)
        assert "d.category = 'stock'" in panel_sql and "d.timestamps < addDays(toDate('2026-11-16'), 1)" in panel_sql
        index = pd.MultiIndex.from_product([dates, stock_codes], names=["date", "code"])
        frame = index.to_frame(index=False)
        t = np.repeat(np.arange(len(dates)), len(stock_codes))
        offset = np.tile(np.arange(len(stock_codes)), len(dates)) * 10.0
        for field in fields:
            frame[field] = 100.0 + t + offset if field in {"open", "close", "high", "low"} else 1.0
        frame["ycp"] = 99.0 + t + offset
        if all_stocks:
            # 第501只股票中途上市。全市场模式保留它，上市前价格和标签为缺失值。
            frame = frame[(frame["code"] != codes[-1]) | (frame["date"] >= dates[6])]
        frame.to_parquet(output, index=False)

    monkeypatch.setattr(clickhouse, "resolve_settings", lambda: clickhouse.LiveDBSettings(password="test"))
    monkeypatch.setattr(clickhouse.ClickHouseHTTPClient, "query", query)
    monkeypatch.setattr(company_source, "write_pool_panel", capture)
    monkeypatch.setattr(comparison_data, "latest_completed_session", lambda: "2026-11-16")
    output, root = tmp_path / "panel", tmp_path / "snapshots"
    argv = ["comparison_cli", "prepare-panel", "--output", str(output), "--snapshot-root", str(root)]
    if not all_stocks:
        argv += ["--codes", str(codes_path)]
    monkeypatch.setattr(sys, "argv", argv)
    if missing_day:
        with pytest.raises(ValueError, match="missing whole-market sessions"):
            comparison_cli.main()
        assert not output.exists()
        return
    comparison_cli.main()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "clickhouse_snapshot"
    assert manifest["date_end"] == "2026-11-16"
    assert manifest["experiment_training_start"] == "2025-07-01"
    assert manifest["source_snapshot"]["as_of_kind"] == "capture-label"
    assert manifest["source_snapshot"]["category"] == "stock"
    assert manifest["stock_universe"] == ("all_database_stocks" if all_stocks else "explicit_codes")
    assert manifest["n_stocks"] == len(codes)
    assert "password" not in json.dumps(manifest).lower()
    dates = np.load(output / "dates.npy")
    labels = np.load(output / manifest["label_file"])
    assert np.isfinite(labels[:-20, :-1]).all()
    if all_stocks:
        assert np.isnan(labels[:6, -1]).all()
        assert np.isnan(np.load(output / "close.npy")[:6, -1]).all()
        assert np.isfinite(labels[6:-20, -1]).all()
    assert np.isnan(labels[-20:]).all()
    assert labels[0, 0] == pytest.approx(120 / 100 - 1)
    assert np.searchsorted(dates, np.datetime64("2025-07-01")) == 100
    assert len(queries) == (2 if all_stocks else 1)
    assert len(captures) == 1

    def offline():
        """离线复用时，配置和查询均不应再次触发。"""
        raise AssertionError("offline load attempted a database connection")

    monkeypatch.setattr(clickhouse, "resolve_settings", offline)
    copy = tmp_path / "offline"
    source_codes = None if all_stocks else codes_path
    comparison_data.prepare_database_panel(source_codes, copy, snapshot_id=manifest["source_snapshot_id"], snapshot_root=root)
    assert np.array_equal(np.load(copy / "close.npy"), np.load(output / "close.npy"), equal_nan=True)
    tasks, _ = comparison_data.design_tasks(copy)
    comparison_data.validate_cache(copy, tasks, 100, 20, comparison_data.FIELDS)
    from single_factor.hierarchical.comparison_study import prepare_protocol
    protocol = prepare_protocol(copy, tmp_path / "protocol", profile="smoke", device="cpu")
    assert protocol["dates"]["training_information_start"] == "2025-07-01"
    assert protocol["dates"]["training_label_cutoff"] == "2026-11-16"
    assert protocol["coverage"]["end"] == "2026-11-16"
    assert len(protocol["config"]["tasks"]) == 19
    with pytest.raises(FileExistsError):
        comparison_data.prepare_database_panel(codes_path, output, snapshot_id=manifest["source_snapshot_id"], snapshot_root=root)
    path = snapshot_store.snapshot_path(manifest["source_snapshot_id"], snapshot_root=root)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        comparison_data.prepare_database_panel(codes_path, tmp_path / "corrupt", snapshot_id=manifest["source_snapshot_id"], snapshot_root=root)
    assert not (tmp_path / "corrupt").exists()
