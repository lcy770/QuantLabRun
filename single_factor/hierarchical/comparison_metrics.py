"""固定成员配权对照与独立 Top50 净收益评价，结果不进入策略奖励。"""
from __future__ import annotations

import numpy as np


def prediction_rank_ic(prediction, target):
    """使用共同有效股票计算组合日 Spearman 均值。"""
    import torch
    from .._vendor.alphagen.utils.correlation import batch_spearmanr
    values = batch_spearmanr(torch.tensor(prediction), torch.tensor(target)).numpy()
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def locked_ridge_weights(evaluator, candidates, selected, ridge=0.001):
    """只在 F 段拟合固定成员的 Ridge 权重，后续段不重估方向或系数。"""
    if not selected:
        return []
    scoring = evaluator.panels["fit"]
    signals = np.stack([evaluator.segment(candidates[i].tree, "fit")[1] * candidates[i].direction for i in selected], axis=-1)
    weights = scoring.fit_weights
    gram = np.einsum("dsi,dsj,ds->ij", signals, signals, weights)
    rhs = np.einsum("dsi,ds,ds->i", signals, scoring.y, weights)
    result = np.linalg.solve(gram + ridge * np.eye(len(selected)), rhs)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite Ridge coefficients")
    return result.tolist()


def prediction_for(evaluator, candidates, selected, segment, weights=None):
    """按提前固定的方向及权重合成每日信号。"""
    scoring = evaluator.panels[segment]
    if not selected:
        return np.zeros_like(scoring.target)
    coefficients = np.full(len(selected), 1 / len(selected)) if weights is None else np.asarray(weights)
    return sum(evaluator.segment(candidates[i].tree, segment)[1] * candidates[i].direction * weight
               for i, weight in zip(selected, coefficients))


def portfolio_curve(scoring, signal, settings, *, benchmark=False, gross=False):
    """收盘排序、次日开盘执行、实际成交日记账，按交易日间隔调仓。"""
    back = scoring.panel.max_backtrack_days
    arrays = {name: np.asarray(values[back:], dtype=float) for name, values in scoring.panel._arrays.items()}
    opening, closing = arrays["open"], arrays["close"]
    dates = scoring.panel.dates[back:]
    cash = initial = float(settings.get("initial_cash", 100_000_000.0))
    shares = np.zeros(opening.shape[1])
    marked = np.zeros_like(shares)
    pending = None
    previous = initial
    rows, orders, fills = [], 0, 0
    fee_rate = 0.0 if gross else settings.get("fee_rate", 0.0015)
    slippage = 0.0 if gross else settings.get("slippage", 0.0005)
    minimum = 0.0 if gross else settings.get("minimum_fee", 5.0)
    top = opening.shape[1] if benchmark else min(settings.get("top_k", 50), opening.shape[1])
    interval = settings.get("rebalance_days", 20)
    for day in range(len(opening)):
        valid_open = np.isfinite(opening[day]) & (opening[day] > 0)
        volume = arrays.get("volume")
        tradable = valid_open.copy()
        if volume is not None:
            tradable &= np.isfinite(volume[day]) & (volume[day] > 0)
        if "Ifsuspend" in arrays:
            tradable &= arrays["Ifsuspend"][day] == 0
        cost, traded = 0.0, 0.0
        if pending is not None:
            at_open = np.where(valid_open, opening[day], marked)
            equity = cash + float(np.dot(shares, at_open))
            target = np.zeros_like(shares)
            target[pending] = equity / max(top, 1) / np.maximum(opening[day, pending], 1e-300)
            sell_allowed = tradable.copy()
            buy_allowed = tradable.copy()
            if "if_down" in arrays:
                sell_allowed &= arrays["if_down"][day] != 1
            if "if_up" in arrays:
                buy_allowed &= arrays["if_up"][day] != 1
            for index in np.flatnonzero(shares > target + 1e-12):
                orders += 1
                if not sell_allowed[index]:
                    continue
                quantity = shares[index] - target[index]
                amount = quantity * opening[day, index] * (1 - slippage)
                fee = max(amount * fee_rate, minimum)
                if cash + amount < fee:
                    continue
                shares[index] -= quantity
                cash += amount - fee
                cost += fee
                traded += amount
                fills += 1
            for index in pending:
                quantity = max(target[index] - shares[index], 0.0)
                if quantity <= 1e-12:
                    continue
                orders += 1
                if not buy_allowed[index]:
                    continue
                price = opening[day, index] * (1 + slippage)
                quantity = min(quantity, max((cash - minimum) / (price * (1 + fee_rate)), 0.0))
                amount = quantity * price
                fee = max(amount * fee_rate, minimum)
                if quantity <= 0 or amount + fee > cash + 1e-7:
                    continue
                shares[index] += quantity
                cash -= amount + fee
                marked[index] = opening[day, index]
                cost += fee
                traded += amount
                fills += 1
        valid_close = np.isfinite(closing[day]) & (closing[day] > 0)
        marked[valid_close] = closing[day, valid_close]
        equity = cash + float(np.dot(shares, marked))
        if not np.isfinite(equity) or cash < -1e-5:
            raise FloatingPointError("invalid portfolio accounting")
        rows.append({"date": str(dates[day].date()), "nav": equity / initial,
                     "return": equity / previous - 1, "turnover": traded / previous,
                     "commission": cost / previous, "cash_weight": cash / max(equity, 1e-300)})
        previous = equity
        pending = None
        if day < len(signal) and day % interval == 0:
            valid = np.isfinite(signal[day])
            if benchmark:
                pending = np.arange(opening.shape[1])
            elif valid.any() and np.ptp(signal[day, valid]) > 1e-12:
                pending = np.lexsort((np.arange(len(shares)), -np.where(valid, signal[day], -np.inf)))[:top]
                pending = pending[valid[pending]]
    nav = np.array([row["nav"] for row in rows])
    return {"net_return": float(nav[-1] - 1), "maximum_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
            "turnover": float(sum(row["turnover"] for row in rows)), "orders": orders,
            "filled_orders": fills, "fill_ratio": fills / max(orders, 1), "daily": rows}


def top50_metrics(evaluator, candidates, selected, segment, settings, weights=None):
    """同一股票池、调仓时点和费用下评价 Top50 与等权股票池基准。"""
    signal = prediction_for(evaluator, candidates, selected, segment, weights)
    net = portfolio_curve(evaluator.panels[segment], signal, settings)
    gross = portfolio_curve(evaluator.panels[segment], signal, settings, gross=True)
    benchmark = portfolio_curve(evaluator.panels[segment], signal, settings, benchmark=True)
    flags = evaluator.panels[segment].panel.feature_names
    return {**net, "gross_return": gross["net_return"], "benchmark_net_return": benchmark["net_return"],
            "net_excess_return": net["net_return"] - benchmark["net_return"],
            "benchmark": "same_universe_equal_weight_net", "timing": "close_signal_next_open_execution",
            "rebalance_trading_days": settings.get("rebalance_days", 20),
            "available_execution_flags": [name for name in ("Ifsuspend", "if_up", "if_down") if name in flags],
            "adjustment": "cache_price_return_convention", "reward_uses_portfolio_return": False}
