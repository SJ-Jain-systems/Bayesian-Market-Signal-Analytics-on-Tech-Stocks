-- Build an analysis-ready price table from raw source records.
-- The cleaned table keeps positive OHLCV observations, standardizes the adjusted
-- close column name, and keeps one record for each symbol/date pair.
CREATE OR REPLACE TABLE clean_prices AS
WITH valid_prices AS (
    SELECT
        symbol,
        date,
        open,
        high,
        low,
        close,
        close_adjusted AS adj_close,
        volume,
        split_coefficient,
        ROW_NUMBER() OVER (
            PARTITION BY symbol, date
            ORDER BY symbol, date
        ) AS duplicate_rank
    FROM raw_prices
    WHERE open > 0
      AND high > 0
      AND low > 0
      AND close > 0
      AND close_adjusted > 0
      AND volume > 0
)
SELECT
    symbol,
    date,
    open,
    high,
    low,
    close,
    adj_close,
    volume,
    split_coefficient
FROM valid_prices
WHERE duplicate_rank = 1
ORDER BY symbol, date;

CREATE OR REPLACE VIEW data_quality_summary AS
SELECT
    symbol,
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    COUNT(*) AS row_count,
    COUNT(DISTINCT date) AS distinct_date_count,
    AVG(volume) AS average_volume,
    MIN(adj_close) AS min_adj_close,
    MAX(adj_close) AS max_adj_close
FROM clean_prices
GROUP BY symbol
ORDER BY symbol;
