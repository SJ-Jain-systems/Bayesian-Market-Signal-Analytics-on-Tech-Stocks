from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, load_raw_stock_data


def _valid_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": [1, 2],
            "symbol": ["MSFT", "AAPL"],
            "date": ["2024-01-02", "2024-01-03"],
            "open": [100.0, 200.0],
            "high": [101.0, 202.0],
            "low": [99.0, 198.0],
            "close": [100.5, 201.0],
            "close_adjusted": [100.5, 201.0],
            "volume": [1_000_000, 2_000_000],
            "split_coefficient": [1.0, 1.0],
        }
    )


def test_load_raw_stock_data_drops_unnamed_index_column(tmp_path):
    csv_path = tmp_path / "raw_prices.csv"
    _valid_raw_data().to_csv(csv_path, index=False)

    result = load_raw_stock_data(csv_path)

    assert "Unnamed: 0" not in result.columns


def test_load_raw_stock_data_converts_date_to_datetime(tmp_path):
    csv_path = tmp_path / "raw_prices.csv"
    _valid_raw_data().to_csv(csv_path, index=False)

    result = load_raw_stock_data(csv_path)

    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_load_raw_stock_data_accepts_all_required_columns(tmp_path):
    csv_path = tmp_path / "raw_prices.csv"
    _valid_raw_data().to_csv(csv_path, index=False)

    result = load_raw_stock_data(csv_path)

    assert set(REQUIRED_COLUMNS).issubset(result.columns)


def test_load_raw_stock_data_raises_for_missing_required_columns(tmp_path):
    csv_path = tmp_path / "raw_prices.csv"
    _valid_raw_data().drop(columns=["close_adjusted"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column.*close_adjusted"):
        load_raw_stock_data(csv_path)
