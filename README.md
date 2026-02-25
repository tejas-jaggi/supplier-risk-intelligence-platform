**Supplier Risk Intelligence Platform**



**Live App:**

https://supplier-risk-intelligence.streamlit.app



An end-to-end machine learning system that predicts late deliveries, order cancellations, and profit margin risk — aggregated into a composite supplier risk score for operational decision-making.



**Business Problem**



Supply chain organizations struggle with:



* Late deliveries impacting SLA compliance
* Order cancellations reducing revenue reliability
* Low-margin orders increasing financial risk
* Lack of a unified supplier risk visibility framework



This platform builds predictive models and aggregates them into a composite risk score to enable:



* Proactive supplier monitoring
* Risk-based procurement prioritization
* Margin protection strategies
* Executive-level supplier risk dashboards



**System Architectur**e



Raw Orders (180K rows)

        ↓

Feature Engineering (behavioral + historical)

        ↓

Model 1 → Late Delivery Classifier (XGBoost)

Model 2 → Cancellation Risk Classifier (XGBoost)

Model 3 → Profit Risk Model

        ↓

Weighted Composite Risk Score

        ↓

Supplier-Level Aggregation

        ↓

Streamlit Risk Intelligence Dashboard



**Models \& Performance**

1️⃣ Late Delivery Classifier



* Algorithm: XGBoost
* ROC-AUC: ~0.73
* Optimized for balanced recall \& precision
* Feature importance validated with SHAP



2️⃣ Cancellation / SLA Breach Model



* Handles severe class imbalance (~5.6% positive class)
* ROC-AUC: ~0.82
* Threshold tuning for recall-focused risk detection



3️⃣ Profit Risk Model



* Profit margin outlier clipping
* Engineered profit risk score
* Aggregated into supplier-level financial exposure signal



**Composite Risk Score**



Final supplier risk score combines:



* 50% Late Delivery Risk
* 30% Cancellation Risk
* 20% Profit Margin Risk



Suppliers are categorized into:



🔴 Critical

🟠 High

🟡 Moderate

🟢 Low



**Dashboard Capabilities**



* Executive risk tier summary
* Supplier rankings
* Department risk heatmap
* Risk contribution breakdown
* Drill-down analytics by category
* Downloadable risk report (CSV)



**Tech Stack**



* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib / Seaborn
* Streamlit
* Git / GitHub



**Dataset**



DataCo Smart Supply Chain Dataset

180,519 orders (2015–2017)



Note: Raw dataset excluded from repository for size compliance.



**Deployment**



The application is deployed via Streamlit Cloud and auto-builds from the main branch.



**Why This Project Matters**



This project demonstrates:



* End-to-end ML system design
* Class imbalance handling
* Feature engineering strategy
* Multi-model aggregation
* Business-aligned scoring logic
* Production deployment workflow
* Dashboard UX for executive stakeholders



**Author**



Tejas Jaggi

Machine Learning \& Supply Chain Analytics

