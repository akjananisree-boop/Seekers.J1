"""
Decision Intelligence Dashboard - Exasol AI Build Challenge 2026

Run with:
    streamlit run app.py

Reads the CSV outputs produced by scripts/forecast_model.py
(data/retail_sales.csv, forecast_output.csv, recommendations.csv,
model_metrics.csv). In production these come from Exasol Personal
via the VW_* views defined in sql/01_schema.sql.
"""

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Demand Forecasting - Exasol AI Build Challenge", layout="wide")

DATA_DIR = "../data"


@st.cache_data
def load_all():
    sales = pd.read_csv(f"{DATA_DIR}/retail_sales.csv", parse_dates=["date"])
    forecast = pd.read_csv(f"{DATA_DIR}/forecast_output.csv", parse_dates=["forecast_date"])
    recs = pd.read_csv(f"{DATA_DIR}/recommendations.csv")
    metrics = pd.read_csv(f"{DATA_DIR}/model_metrics.csv")
    return sales, forecast, recs, metrics


sales, forecast, recs, metrics = load_all()

st.title("📊 AI-Powered Demand Forecasting")
st.caption("Exasol AI Build Challenge 2026 · Decision Intelligence Track · Data platform: Exasol Personal")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
store_options = sorted(sales["store_id"].unique())
store_id = st.sidebar.selectbox("Store", store_options)

product_options = sorted(sales[sales["store_id"] == store_id]["product_id"].unique())
product_names = sales[sales["product_id"].isin(product_options)][["product_id", "product_name"]].drop_duplicates()
product_id = st.sidebar.selectbox(
    "Product",
    product_options,
    format_func=lambda pid: product_names.set_index("product_id").loc[pid, "product_name"],
)

# ---------------- Top KPI row ----------------
total_revenue = sales["revenue_inr"].sum()
total_units = sales["units_sold"].sum()
avg_mae_pct = metrics["mae_pct"].mean()
stockout_days = sales["stockout_flag"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue (2yr)", f"₹{total_revenue/1e5:.1f}L")
k2.metric("Total Units Sold", f"{total_units:,.0f}")
k3.metric("Avg Forecast Error (MAE%)", f"{avg_mae_pct:.1f}%")
k4.metric("Stockout Days Recorded", f"{stockout_days:,.0f}")

st.divider()

# ---------------- Forecast chart ----------------
st.subheader(f"14-Day Forecast — {product_names.set_index('product_id').loc[product_id, 'product_name']} @ {store_id}")

hist = sales[(sales["store_id"] == store_id) & (sales["product_id"] == product_id)].copy()
hist_daily = hist.groupby("date", as_index=False)["units_sold"].sum().tail(60)
hist_daily["type"] = "Actual"
hist_daily = hist_daily.rename(columns={"date": "day", "units_sold": "units"})

fc = forecast[(forecast["store_id"] == store_id) & (forecast["product_id"] == product_id)].copy()
fc = fc.rename(columns={"forecast_date": "day", "forecast_units": "units"})
fc["type"] = "Forecast"

combined = pd.concat([hist_daily[["day", "units", "type"]], fc[["day", "units", "type"]]])

fig = px.line(combined, x="day", y="units", color="type", markers=True,
              color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"})
fig.update_layout(height=420, legend_title="", xaxis_title="", yaxis_title="Units")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------- Recommendations table ----------------
st.subheader("🎯 Reorder Recommendations (Decision Intelligence Output)")
st.dataframe(
    recs.sort_values("recommended_reorder_qty", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------- Category breakdown ----------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Revenue by Category")
    cat_rev = sales.groupby("category", as_index=False)["revenue_inr"].sum()
    fig2 = px.pie(cat_rev, names="category", values="revenue_inr", hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Model Accuracy by Store/Product")
    fig3 = px.bar(metrics.sort_values("mae_pct"), x="product_id", y="mae_pct", color="store_id", barmode="group")
    fig3.update_layout(yaxis_title="MAE %", xaxis_title="Product")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption(
    "Primary data platform: **Exasol Personal**. Raw sales land in SALES_RAW; "
    "in-database views (VW_DAILY_DEMAND, VW_WEEKLY_DEMAND, VW_PRODUCT_VELOCITY_30D, "
    "VW_DEMAND_ANOMALIES) do the heavy aggregation, which the Python AI layer then "
    "reads to train per-product forecasting models and generate reorder recommendations."
)
