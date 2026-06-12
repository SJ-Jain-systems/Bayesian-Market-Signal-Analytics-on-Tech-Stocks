-- Raw technology equity OHLCV price history loaded from source files/APIs.
-- This table intentionally preserves the source-level schema with minimal typing so
-- downstream scripts can apply repeatable validation, de-duplication, and feature
-- engineering steps in DuckDB.
CREATE TABLE IF NOT EXISTS raw_prices (
    symbol VARCHAR,
    date DATE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    close_adjusted DOUBLE,
    volume BIGINT,
    split_coefficient DOUBLE
);

COMMENT ON TABLE raw_prices IS
    'Source OHLCV price records for technology stocks before cleaning or de-duplication.';
