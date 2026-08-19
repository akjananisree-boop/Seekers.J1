# Pitch Deck Outline — AI-Powered Demand Forecasting

Use this as your slide-by-slide script. Aim for ~10 slides, 3-minute demo video.

---

**Slide 1 — Title**
AI-Powered Demand Forecasting
Decision Intelligence Track · Exasol AI Build Challenge 2026
Team: [JANANISREE A K,KALAIVANI G,MANJASHREE P A,MADHUMITHRA A]

**Slide 2 — The Problem**
- Retailers over-order slow movers, stock out on fast movers
- Manual reordering is reactive, not predictive
- 2% of our sample data already shows stockout days — real revenue lost

**Slide 3 — Our Solution**
"We forecast demand 14 days ahead per store/product, then turn that
forecast directly into a reorder recommendation — not just a chart."

**Slide 4 — Architecture**
(Insert the architecture diagram from README.md)
Exasol Personal → in-database views → Python ML layer → dashboard

**Slide 5 — Why Exasol**
- Raw sales land in Exasol's SALES_RAW table
- In-database views handle daily/weekly rollups and anomaly detection
  via window functions — fast even as data scales
- The AI layer only touches small, pre-aggregated results, not raw rows

**Slide 6 — The Model**
- Gradient Boosting Regressor per store/product
- Features: day-of-week, month, lag-7/14, rolling 7/14-day means, promo flag
- Validation MAE: ~12–22% across products — competitive for a v1 model

**Slide 7 — Live Demo**
(Switch to the Streamlit dashboard)
- Show forecast chart for one product
- Show reorder recommendation table
- Show anomaly/velocity view

**Slide 8 — Decision Intelligence, Not Just a Forecast**
- Every forecast → a concrete action: "Reorder ~X units before [date]"
- Anomalies flagged in-database, ready for real-time alerting

**Slide 9 — What's Next**
- Real-time anomaly alerts (Slack/email)
- Price-elasticity + promotion-timing recommendations
- Global model with store/product embeddings for faster scale-up

**Slide 10 — Thank You / Q&A**
GitHub repo: [https://github.com/akjananisree-boop/Seekers.J1]
Team: [Seekers]

---

## Demo Video Script (under 3 minutes)

0:00–0:20 — Problem statement (voiceover over Slide 2)
0:20–0:45 — Solution + architecture (Slides 3–5)
0:45–2:15 — Live dashboard walkthrough:
  - Pick a store/product, show forecast chart
  - Scroll reorder recommendations table
  - Point out one anomaly and explain what it means for the business
2:15–2:45 — Why Exasol mattered (in-database views doing the aggregation)
2:45–3:00 — Close: what's next, thank you
