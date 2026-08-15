"""
Generates a realistic synthetic retail sales dataset for the
Exasol AI Build Challenge 2026 - Demand Forecasting project.

Simulates 3 stores x 8 products over 2 years of daily sales,
with trend, weekly seasonality, yearly seasonality, promotions,
stockout events, and noise - so the forecasting model has
real patterns to learn from.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
dates = pd.date_range(START_DATE, END_DATE, freq="D")

stores = [
    {"store_id": "S01", "store_name": "Chennai Central", "region": "South"},
    {"store_id": "S02", "store_name": "Coimbatore Hub", "region": "South"},
    {"store_id": "S03", "store_name": "Bengaluru East", "region": "South"},
]

products = [
    {"product_id": "P01", "product_name": "Instant Coffee 200g", "category": "Beverages", "base_demand": 45},
    {"product_id": "P02", "product_name": "Basmati Rice 5kg", "category": "Staples", "base_demand": 60},
    {"product_id": "P03", "product_name": "Toothpaste 150g", "category": "Personal Care", "base_demand": 35},
    {"product_id": "P04", "product_name": "Cooking Oil 1L", "category": "Staples", "base_demand": 50},
    {"product_id": "P05", "product_name": "Notebook A4 (Pack of 5)", "category": "Stationery", "base_demand": 20},
    {"product_id": "P06", "product_name": "LED Bulb 9W", "category": "Home", "base_demand": 25},
    {"product_id": "P07", "product_name": "Biscuits Family Pack", "category": "Snacks", "base_demand": 55},
    {"product_id": "P08", "product_name": "Detergent Powder 1kg", "category": "Home", "base_demand": 30},
]

rows = []
for store in stores:
    store_factor = np.random.uniform(0.85, 1.2)
    for product in products:
        product_trend = np.random.uniform(0.00005, 0.00025)  # slow daily growth
        yearly_amp = np.random.uniform(0.15, 0.35)
        for i, d in enumerate(dates):
            base = product["base_demand"] * store_factor

            # long-term growth trend
            trend = 1 + product_trend * i

            # yearly seasonality (peak around Oct-Dec festive season in India)
            day_of_year = d.timetuple().tm_yday
            yearly_season = 1 + yearly_amp * np.sin(2 * np.pi * (day_of_year - 260) / 365)

            # weekly seasonality (weekends higher for most categories)
            weekly_season = 1.25 if d.weekday() in (5, 6) else 1.0

            # festive spikes: Diwali-ish window (mid Oct - mid Nov) and New Year
            festive_boost = 1.0
            if (d.month == 10 and d.day >= 15) or (d.month == 11 and d.day <= 15):
                festive_boost = 1.6
            if d.month == 12 and d.day >= 25:
                festive_boost = 1.4

            # random promotion days (~5% of days), boosts demand ~40-80%
            promo = np.random.rand() < 0.05
            promo_boost = np.random.uniform(1.4, 1.8) if promo else 1.0

            # noise
            noise = np.random.normal(1.0, 0.12)

            demand = base * trend * yearly_season * weekly_season * festive_boost * promo_boost * noise
            demand = max(0, demand)

            # occasional stockout: recorded units_sold capped below true demand
            stockout = np.random.rand() < 0.02
            units_sold = int(round(demand * (np.random.uniform(0.3, 0.6) if stockout else 1.0)))

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "store_id": store["store_id"],
                "store_name": store["store_name"],
                "region": store["region"],
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "category": product["category"],
                "units_sold": units_sold,
                "on_promotion": int(promo),
                "stockout_flag": int(stockout),
                "unit_price_inr": round(np.random.uniform(0.9, 1.1) * {"P01":220,"P02":650,"P03":95,"P04":180,"P05":150,"P06":120,"P07":60,"P08":210}[product["product_id"]], 2),
            })

df = pd.DataFrame(rows)
df["revenue_inr"] = (df["units_sold"] * df["unit_price_inr"]).round(2)
df = df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

out_path = "/home/claude/exasol-demand-forecast/data/retail_sales.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} rows -> {out_path}")
print(df.head())
