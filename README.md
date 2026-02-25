# 🔴 Supplier Risk & Cost Escalation Prediction

**A machine learning system to predict late deliveries, SLA breaches, order cancellations, and profit overruns — aggregated into a composite supplier risk score.**

---

## 📁 Project Structure

```
supplier_risk_project/
├── data/
│   └── DataCoSupplyChainDataset.csv        ← Raw dataset (180K orders)
├── notebooks/
│   ├── 01_EDA.py                           ← Day 1-2: Exploratory analysis
│   ├── 02_feature_engineering.py           ← Day 4-5: Feature creation
│   ├── 03_model_late_delivery.py           ← Day 6-8: Late delivery classifier
│   ├── 04_model_cancellation.py            ← Day 9: Cancellation/SLA model
│   ├── 05_model_profit_overrun.py          ← Day 10: Profit regression model
│   └── 06_risk_scoring.py                  ← Day 11-12: Composite risk scores
├── src/
│   ├── features.py                         ← Reusable feature engineering
│   └── risk_scorer.py                      ← Supplier risk scoring logic
├── models/                                 ← Saved .pkl model files
├── outputs/                                ← Charts, CSVs, reports
├── app.py                                  ← Streamlit dashboard (Day 12-13)
└── requirements.txt
```

---

## 🎯 Prediction Targets

| Model | Target | Type | Metric |
|---|---|---|---|
| Late Delivery | `Late_delivery_risk` | Binary Classification | F1, ROC-AUC |
| SLA / Cancellation | `Order Status` (CANCELED/FRAUD) | Binary Classification | F1, Precision |
| Profit Overrun | `Order Item Profit Ratio` | Regression | RMSE, R² |
| Risk Score | Composite (weighted model outputs) | Scoring | Business KPI |

---

## ⚙️ Setup

```bash
git clone <your-repo>
cd supplier_risk_project
pip install -r requirements.txt

# Place DataCoSupplyChainDataset.csv in data/
python notebooks/01_EDA.py
```

---

## 📊 Key Findings (EDA)

- **54.8%** of all orders experience late delivery
- Standard Class shipping has the highest late rate despite being the most used mode
- Tight average profit margins (~17%) make discount-heavy orders high-risk
- LATAM and Africa regions show elevated late delivery + low profit combinations

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `SHAP` · `Matplotlib/Seaborn` · `Streamlit`

---

## 📅 Build Timeline

| Days | Milestone |
|---|---|
| 1–3 | EDA & data profiling |
| 4–5 | Feature engineering |
| 6–8 | Late delivery model (anchor model) |
| 9–10 | Cancellation & profit models |
| 11–12 | Supplier risk scoring & dashboard |
| 13 | SHAP explainability |
| 14–15 | Polish, README, GitHub |

---

*Dataset: DataCo Smart Supply Chain (Constante et al., 2019) — 180,519 orders, 2015–2017*
