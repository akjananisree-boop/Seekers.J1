-- ==========================================================
-- Exasol AI Build Challenge 2026 - Demand Forecasting
-- Schema: raw sales table + aggregated views
-- Run this against your Exasol Personal instance
-- ==========================================================

CREATE SCHEMA IF NOT EXISTS DEMAND_FORECAST;
OPEN SCHEMA DEMAND_FORECAST;

-- Raw fact table: one row per store/product/day
CREATE OR REPLACE TABLE SALES_RAW (
    SALE_DATE       DATE            NOT NULL,
    STORE_ID        VARCHAR(10)     NOT NULL,
    STORE_NAME      VARCHAR(100),
    REGION          VARCHAR(50),
    PRODUCT_ID      VARCHAR(10)     NOT NULL,
    PRODUCT_NAME    VARCHAR(100),
    CATEGORY        VARCHAR(50),
    UNITS_SOLD      DECIMAL(10,2),
    ON_PROMOTION    BOOLEAN,
    STOCKOUT_FLAG   BOOLEAN,
    UNIT_PRICE_INR  DECIMAL(10,2),
    REVENUE_INR     DECIMAL(14,2)
);

-- ==========================================================
-- Decision-Intelligence layer: pre-aggregated views
-- Exasol's in-database analytics do the heavy lifting here,
-- so the AI/forecasting layer only reads small, ready-to-use
-- aggregates instead of scanning raw rows every time.
-- ==========================================================

-- Daily demand per product per store (model training input)
CREATE OR REPLACE VIEW VW_DAILY_DEMAND AS
SELECT
    SALE_DATE,
    STORE_ID,
    PRODUCT_ID,
    PRODUCT_NAME,
    CATEGORY,
    SUM(UNITS_SOLD)                                   AS TOTAL_UNITS,
    SUM(REVENUE_INR)                                  AS TOTAL_REVENUE,
    MAX(ON_PROMOTION::INT)                            AS HAD_PROMOTION,
    MAX(STOCKOUT_FLAG::INT)                           AS HAD_STOCKOUT
FROM SALES_RAW
GROUP BY SALE_DATE, STORE_ID, PRODUCT_ID, PRODUCT_NAME, CATEGORY;

-- Weekly rollup per product (trend view for dashboard)
CREATE OR REPLACE VIEW VW_WEEKLY_DEMAND AS
SELECT
    STORE_ID,
    PRODUCT_ID,
    PRODUCT_NAME,
    CATEGORY,
    DATE_TRUNC('WEEK', SALE_DATE)                     AS WEEK_START,
    SUM(UNITS_SOLD)                                   AS WEEKLY_UNITS,
    SUM(REVENUE_INR)                                  AS WEEKLY_REVENUE
FROM SALES_RAW
GROUP BY STORE_ID, PRODUCT_ID, PRODUCT_NAME, CATEGORY, DATE_TRUNC('WEEK', SALE_DATE);

-- Fast-moving / slow-moving product ranking (last 30 days)
CREATE OR REPLACE VIEW VW_PRODUCT_VELOCITY_30D AS
SELECT
    PRODUCT_ID,
    PRODUCT_NAME,
    CATEGORY,
    SUM(UNITS_SOLD)                                   AS UNITS_LAST_30D,
    RANK() OVER (ORDER BY SUM(UNITS_SOLD) DESC)        AS DEMAND_RANK
FROM SALES_RAW
WHERE SALE_DATE >= (SELECT MAX(SALE_DATE) FROM SALES_RAW) - 30
GROUP BY PRODUCT_ID, PRODUCT_NAME, CATEGORY;

-- Anomaly detection helper: flags days where units sold deviate
-- more than 2x the trailing 14-day average (simple z-score-ish rule,
-- refined further by the Python model in the AI layer)
CREATE OR REPLACE VIEW VW_DEMAND_ANOMALIES AS
SELECT
    d.SALE_DATE,
    d.STORE_ID,
    d.PRODUCT_ID,
    d.PRODUCT_NAME,
    d.TOTAL_UNITS,
    AVG(d.TOTAL_UNITS) OVER (
        PARTITION BY d.STORE_ID, d.PRODUCT_ID
        ORDER BY d.SALE_DATE
        ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
    ) AS TRAILING_14D_AVG
FROM VW_DAILY_DEMAND d
QUALIFY TOTAL_UNITS > 2 * TRAILING_14D_AVG
   OR TOTAL_UNITS < 0.4 * TRAILING_14D_AVG;
