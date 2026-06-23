from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    create_momentum_features,
    create_return_features,
    create_rolling_features,
    create_trend_features,
)


def test_create_return_features_calculates_simple_and_log_returns():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "adj_close": [100.0, 110.0, 99.0],
            "volume": [10, 20, 30],
        }
    )

    result = create_return_features(df)

    assert pd.isna(result.loc[0, "simple_return"])
    assert result.loc[1, "simple_return"] == pytest.approx(0.10)
    assert result.loc[2, "simple_return"] == pytest.approx(-0.10)
    assert result.loc[1, "log_return"] == pytest.approx(np.log(1.10))
    assert result.loc[2, "log_return"] == pytest.approx(np.log(0.90))


def test_create_return_features_calculates_dollar_volume():
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "adj_close": [100.0, 110.0],
            "volume": [10, 20],
        }
    )

    result = create_return_features(df)

    assert result["dollar_volume"].tolist() == [1000.0, 2200.0]


def test_create_rolling_features_are_grouped_by_symbol():
    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 21 + ["MSFT"] * 21,
            "date": list(dates) + list(dates),
            "simple_return": [0.01] * 21 + [0.02] * 21,
        }
    )

    result = create_rolling_features(df)
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    msft = result[result["symbol"] == "MSFT"].reset_index(drop=True)

    assert aapl.loc[:19, "rolling_mean_21d"].isna().all()
    assert msft.loc[:19, "rolling_mean_21d"].isna().all()
    assert aapl.loc[20, "rolling_mean_21d"] == pytest.approx(0.01)
    assert msft.loc[20, "rolling_mean_21d"] == pytest.approx(0.02)


def test_create_momentum_features_calculates_21d_return():
    dates = pd.date_range("2024-01-01", periods=22, freq="D")
    prices = [100.0 + i for i in range(22)]
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 22,
            "date": dates,
            "adj_close": prices,
        }
    )

    result = create_momentum_features(df)

    assert result.loc[:20, "return_21d"].isna().all()
    assert result.loc[21, "return_21d"] == pytest.approx(prices[21] / prices[0] - 1.0)


def test_create_trend_features_ma_20_requires_full_window():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = [100.0 + i for i in range(20)]
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 20,
            "date": dates,
            "adj_close": prices,
        }
    )

    result = create_trend_features(df)

    assert result.loc[:18, "ma_20"].isna().all()
    assert result.loc[19, "ma_20"] == pytest.approx(sum(prices) / len(prices))
