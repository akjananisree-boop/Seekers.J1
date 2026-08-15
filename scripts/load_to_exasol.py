"""
Loads data/retail_sales.csv into an Exasol Personal instance.

Prereqs:
  pip install pyexasol pandas

Exasol Personal connection defaults (adjust as needed):
  - Local Docker install : host=localhost, port=8563
  - AWS/Azure deployment  : use the host/IP given by your deployment output

Usage:
  python load_to_exasol.py --host localhost --port 8563 --user sys --password exasol
"""

import argparse
import pandas as pd
import pyexasol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8563, type=int)
    parser.add_argument("--user", default="sys")
    parser.add_argument("--password", default="exasol")
    parser.add_argument("--csv", default="../data/retail_sales.csv")
    args = parser.parse_args()

    print(f"Connecting to Exasol at {args.host}:{args.port} ...")
    conn = pyexasol.connect(
        dsn=f"{args.host}:{args.port}",
        user=args.user,
        password=args.password,
        compression=True,
    )

    print("Running schema script (01_schema.sql) ...")
    with open("../sql/01_schema.sql", "r") as f:
        schema_sql = f.read()
    for stmt in schema_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    print("Loading CSV into SALES_RAW via pyexasol import (fast path) ...")
    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df.rename(columns={
        "date": "SALE_DATE",
        "store_id": "STORE_ID",
        "store_name": "STORE_NAME",
        "region": "REGION",
        "product_id": "PRODUCT_ID",
        "product_name": "PRODUCT_NAME",
        "category": "CATEGORY",
        "units_sold": "UNITS_SOLD",
        "on_promotion": "ON_PROMOTION",
        "stockout_flag": "STOCKOUT_FLAG",
        "unit_price_inr": "UNIT_PRICE_INR",
        "revenue_inr": "REVENUE_INR",
    })
    df["ON_PROMOTION"] = df["ON_PROMOTION"].astype(bool)
    df["STOCKOUT_FLAG"] = df["STOCKOUT_FLAG"].astype(bool)

    conn.execute("DELETE FROM DEMAND_FORECAST.SALES_RAW")
    conn.import_from_pandas(df, ("DEMAND_FORECAST", "SALES_RAW"))

    row_count = conn.execute("SELECT COUNT(*) FROM DEMAND_FORECAST.SALES_RAW").fetchone()[0]
    print(f"Loaded {row_count:,} rows into DEMAND_FORECAST.SALES_RAW")

    conn.close()


if __name__ == "__main__":
    main()
