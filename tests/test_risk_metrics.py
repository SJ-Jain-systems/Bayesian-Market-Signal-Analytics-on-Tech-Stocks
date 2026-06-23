from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    annualized_return,
    annualized_volatility,
    bad_day_rate,
    conditional_value_at_risk,
    downside_deviation,
    max_drawdown,
    sharpe_like_ratio,
    value_at_risk,
)


def test_annualized_return_known_values():
    returns = np.array([0.01, 0.02, -0.01])

    assert annualized_return(returns, trading_days=10) == pytest.approx(
        returns.mean() * 10
    )


def test_annualized_volatility_known_values():
    returns = np.array([0.01, 0.02, -0.01])

    assert annualized_volatility(returns, trading_days=10) == pytest.approx(
        returns.std(ddof=1) * np.sqrt(10)
    )


def test_max_drawdown_known_price_path():
    prices = [100.0, 120.0, 90.0, 130.0, 104.0]

    assert max_drawdown(prices) == pytest.approx(-0.25)


def test_value_at_risk_known_quantile():
    returns = pd.Series([-0.05, -0.02, 0.0, 0.01])

    assert value_at_risk(returns, alpha=0.25) == pytest.approx(returns.quantile(0.25))


def test_bad_day_rate_counts_returns_at_or_below_threshold():
    returns = [-0.03, -0.02, 0.01, np.nan, np.inf]

    assert bad_day_rate(returns, threshold=-0.02) == pytest.approx(2 / 3)


def test_conditional_value_at_risk_known_values():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.03])

    var_threshold = returns.quantile(0.4)
    expected = returns[returns <= var_threshold].mean()

    assert conditional_value_at_risk(returns, alpha=0.4) == pytest.approx(expected)
    assert conditional_value_at_risk(returns, alpha=0.4) == pytest.approx(-0.075)


def test_downside_deviation_known_values():
    returns = np.array([0.02, -0.01, 0.03, -0.04])
    threshold = 0.0

    shortfalls = np.minimum(returns - threshold, 0.0)
    expected = np.sqrt(np.mean(np.square(shortfalls)))

    assert downside_deviation(returns, threshold=threshold) == pytest.approx(expected)


def test_sharpe_like_ratio_known_values():
    returns = np.array([0.01, 0.02, -0.01])
    trading_days = 10
    risk_free_rate = 0.01

    ann_return = annualized_return(returns, trading_days=trading_days)
    ann_volatility = annualized_volatility(returns, trading_days=trading_days)
    expected = (ann_return - risk_free_rate) / ann_volatility

    assert sharpe_like_ratio(
        returns, trading_days=trading_days, risk_free_rate=risk_free_rate
    ) == pytest.approx(expected)
