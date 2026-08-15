"""
Demand Forecasting model for the Exasol AI Build Challenge 2026.

In production this reads from Exasol's VW_DAILY_DEMAND view
(see sql/01_schema.sql) via pyexasol. For local development /
demo purposes it can also read directly from data/retail_sales.csv
if EXASOL_HOST is not set.

Approach:
  - Feature engineering: day-of-week, month, lag-7, lag-14,
    rolling-7 mean, rolling-14 mean, promotion flag
  - Model: Gradient Boosting Regressor, one model per product
    (fast to train, handles non-linear seasonality well)
  - Output: 14-day forecast per store/product + reorder
    recommendations (decision-intelligence layer)

Usage:
  python forecast_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FORECAST_HORIZON_DAYS = 14
LEAD_TIME_DAYS = 5          # assumed supplier lead time for reorder logic
SAFETY_STOCK_DAYS = 3


def load_data():
    exasol_host = os.environ.get("EXASOL_HOST")
    if exasol_host:
        import pyexasol
        conn = pyexasol.connect(
            dsn=f"{exasol_host}:{os.environ.get('EXASOL_PORT', 8563)}",
            user=os.environ.get("EXASOL_USER", "sys"),
            password=os.environ.get("EXASOL_PASSWORD", "exasol"),
        )
        df = conn.export_to_pandas("SELECT * FROM DEMAND_FORECAST.VW_DAILY_DEMAND")
        conn.close()
        df = df.rename(columns={
            "SALE_DATE": "date", "STORE_ID": "store_id", "PRODUCT_ID": "product_id",
            "PRODUCT_NAME": "product_name", "CATEGORY": "category",
            "TOTAL_UNITS": "units_sold", "HAD_PROMOTION": "on_promotion",
        })
    else:
        raw = pd.read_csv("../data/retail_sales.csv", parse_dates=["date"])
        df = (
            raw.groupby(["date", "store_id", "product_id", "product_name", "category"], as_index=False)
            .agg(units_sold=("units_sold", "sum"), on_promotion=("on_promotion", "max"))
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def make_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy().sort_values("date").reset_index(drop=True)
    g["dow"] = g["date"].dt.dayofweek
    g["month"] = g["date"].dt.month
    g["is_weekend"] = g["dow"].isin([5, 6]).astype(int)
    g["lag_7"] = g["units_sold"].shift(7)
    g["lag_14"] = g["units_sold"].shift(14)
    g["roll_7_mean"] = g["units_sold"].shift(1).rolling(7).mean()
    g["roll_14_mean"] = g["units_sold"].shift(1).rolling(14).mean()
    return g


FEATURES = ["dow", "month", "is_weekend", "on_promotion",
            "lag_7", "lag_14", "roll_7_mean", "roll_14_mean"]


def train_and_forecast(df: pd.DataFrame) -> pd.DataFrame:
    all_forecasts = []
    all_metrics = []

    for (store_id, product_id), grp in df.groupby(["store_id", "product_id"]):
        feat = make_features(grp)
        feat_clean = feat.dropna(subset=FEATURES + ["units_sold"])
        if len(feat_clean) < 60:
            continue  # not enough history for this store/product pair

        # simple time-based train/validation split (last 14 days = validation)
        train = feat_clean.iloc[:-14]
        val = feat_clean.iloc[-14:]

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        model.fit(train[FEATURES], train["units_sold"])

        val_pred = model.predict(val[FEATURES])
        mae = mean_absolute_error(val["units_sold"], val_pred)
        avg_actual = val["units_sold"].mean()
        all_metrics.append({
            "store_id": store_id, "product_id": product_id,
            "mae": round(mae, 2), "avg_daily_units": round(avg_actual, 2),
            "mae_pct": round(100 * mae / max(avg_actual, 1e-6), 1),
        })

        # refit on full history, then roll forward day-by-day for the horizon
        model.fit(feat_clean[FEATURES], feat_clean["units_sold"])
        history = feat.copy()
        last_date = history["date"].max()
        product_name = grp["product_name"].iloc[0]
        category = grp["category"].iloc[0]

        for step in range(1, FORECAST_HORIZON_DAYS + 1):
            next_date = last_date + pd.Timedelta(days=step)
            dow = next_date.dayofweek
            month = next_date.month
            is_weekend = int(dow in [5, 6])
            lag_7 = history["units_sold"].iloc[-7] if len(history) >= 7 else history["units_sold"].mean()
            lag_14 = history["units_sold"].iloc[-14] if len(history) >= 14 else history["units_sold"].mean()
            roll_7 = history["units_sold"].iloc[-7:].mean()
            roll_14 = history["units_sold"].iloc[-14:].mean()

            x_row = pd.DataFrame([{
                "dow": dow, "month": month, "is_weekend": is_weekend,
                "on_promotion": 0, "lag_7": lag_7, "lag_14": lag_14,
                "roll_7_mean": roll_7, "roll_14_mean": roll_14,
            }])
            pred = max(0, model.predict(x_row[FEATURES])[0])

            all_forecasts.append({
                "store_id": store_id, "product_id": product_id,
                "product_name": product_name, "category": category,
                "forecast_date": next_date.strftime("%Y-%m-%d"),
                "forecast_units": round(pred, 1),
            })

            history = pd.concat([history, pd.DataFrame([{
                "date": next_date, "store_id": store_id, "product_id": product_id,
                "product_name": product_name, "category": category,
                "units_sold": pred, "on_promotion": 0, "dow": dow, "month": month,
                "is_weekend": is_weekend, "lag_7": lag_7, "lag_14": lag_14,
                "roll_7_mean": roll_7, "roll_14_mean": roll_14,
            }])], ignore_index=True)

    forecast_df = pd.DataFrame(all_forecasts)
    metrics_df = pd.DataFrame(all_metrics)
    return forecast_df, metrics_df


def build_recommendations(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Decision-intelligence layer: turn forecasts into reorder actions."""
    recs = []
    for (store_id, product_id), grp in forecast_df.groupby(["store_id", "product_id"]):
        grp = grp.sort_values("forecast_date")
        avg_daily = grp["forecast_units"].mean()
        lead_time_demand = grp["forecast_units"].iloc[:LEAD_TIME_DAYS].sum()
        safety_stock = avg_daily * SAFETY_STOCK_DAYS
        reorder_point = round(lead_time_demand + safety_stock)
        peak_day = grp.loc[grp["forecast_units"].idxmax()]

        recs.append({
            "store_id": store_id,
            "product_id": product_id,
            "product_name": grp["product_name"].iloc[0],
            "category": grp["category"].iloc[0],
            "avg_forecast_daily_units": round(avg_daily, 1),
            "recommended_reorder_qty": reorder_point,
            "peak_demand_date": peak_day["forecast_date"],
            "peak_demand_units": peak_day["forecast_units"],
            "action": (
                f"Reorder ~{reorder_point} units before {grp['forecast_date'].iloc[LEAD_TIME_DAYS-1]} "
                f"to cover lead time + safety stock."
            ),
        })
    return pd.DataFrame(recs)


def main():
    print("Loading daily demand data ...")
    df = load_data()
    print(f"  {len(df):,} store-product-day rows across "
          f"{df['store_id'].nunique()} stores and {df['product_id'].nunique()} products")

    print("Training per-product models and generating 14-day forecasts ...")
    forecast_df, metrics_df = train_and_forecast(df)

    print("Building reorder recommendations ...")
    recs_df = build_recommendations(forecast_df)

    forecast_df.to_csv("../data/forecast_output.csv", index=False)
    metrics_df.to_csv("../data/model_metrics.csv", index=False)
    recs_df.to_csv("../data/recommendations.csv", index=False)

    print("\n=== Model Accuracy (validation MAE %) ===")
    print(metrics_df.sort_values("mae_pct").to_string(index=False))

    print("\n=== Sample Recommendations ===")
    print(recs_df.head(8).to_string(index=False))

    print("\nSaved: data/forecast_output.csv, data/model_metrics.csv, data/recommendations.csv")


if __name__ == "__main__":
    main()
