"""
=============================================================
SUPPLIER RISK & COST ESCALATION PREDICTION
Final Production Dashboard
=============================================================
Run with: streamlit run app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supplier Risk Monitor",
    page_icon="🔴",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "final_supplier_risk_summary.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Ensure rank exists
    if "risk_rank" not in df.columns:
        df = df.sort_values("composite_risk", ascending=False).reset_index(drop=True)
        df["risk_rank"] = df.index + 1

    # Department summary
    dept = (
        df.groupby("Department Name")
        .agg(
            avg_late_risk=("avg_late_prob", "mean"),
            avg_cancel_risk=("avg_cancel_prob", "mean"),
            avg_profit_risk=("avg_profit_risk", "mean"),
            composite_risk=("composite_risk", "mean"),
        )
        .reset_index()
    )

    return df, dept


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("Supplier Risk Monitor")
st.sidebar.markdown("---")

try:
    supplier_scores, dept_summary = load_data()
    data_loaded = True
except Exception as e:
    st.sidebar.error("Run full pipeline first:")
    st.sidebar.code("python notebooks/05_risk_scoring.py")
    data_loaded = False


# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────
if data_loaded:

    st.title("🔴 Supplier Risk & Cost Escalation Monitor")
    st.caption("Late Delivery • SLA Breach • Margin Risk Aggregation")

    st.markdown("---")

    # KPI METRICS
    col1, col2, col3, col4 = st.columns(4)
    tier_counts = supplier_scores["risk_tier"].value_counts()

    col1.metric("Critical Suppliers", tier_counts.get("Critical", 0))
    col2.metric("High Risk", tier_counts.get("High", 0))
    col3.metric("Medium Risk", tier_counts.get("Medium", 0))
    col4.metric("Low Risk", tier_counts.get("Low", 0))

    st.markdown("---")

    # FILTERS
    all_depts = ["All"] + sorted(supplier_scores["Department Name"].unique())
    sel_dept = st.sidebar.selectbox("Filter by Department", all_depts)

    sel_tier = st.sidebar.multiselect(
        "Risk Tier",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )

    filtered = supplier_scores.copy()

    if sel_dept != "All":
        filtered = filtered[filtered["Department Name"] == sel_dept]

    if sel_tier:
        filtered = filtered[filtered["risk_tier"].isin(sel_tier)]

    # ─────────────────────────────────────────
    # RANKING TABLE + PIE
    # ─────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📋 Supplier Risk Rankings")

        display_cols = [
            "risk_rank",
            "Department Name",
            "Category Name",
            "total_orders",
            "avg_late_prob",
            "avg_cancel_prob",
            "avg_profit_risk",
            "composite_risk",
            "risk_tier"
        ]

        display_df = filtered[display_cols].copy()

        display_df["avg_late_prob"] *= 100
        display_df["avg_cancel_prob"] *= 100
        display_df["avg_profit_risk"] *= 100
        display_df["composite_risk"] *= 100

        display_df.columns = [
            "Rank",
            "Department",
            "Category",
            "Orders",
            "Late Risk (%)",
            "Cancel Risk (%)",
            "Profit Risk (%)",
            "Composite Score (%)",
            "Tier"
        ]

        st.dataframe(display_df, use_container_width=True, height=420)

    with col_right:
        st.subheader("📊 Risk Distribution")

        fig, ax = plt.subplots()
        tc = filtered["risk_tier"].value_counts()
        colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60"]

        ax.pie(tc.values, labels=tc.index, autopct="%1.0f%%", colors=colors)
        ax.set_title("Risk Tier Breakdown")

        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ─────────────────────────────────────────
    # SUPPLIER DRILL-DOWN
    # ─────────────────────────────────────────
    st.subheader("🔎 Supplier Drill-Down Analysis")

    supplier_list = filtered["Category Name"].unique()
    selected_supplier = st.selectbox("Select Supplier Category", supplier_list)

    supplier_row = filtered[filtered["Category Name"] == selected_supplier].iloc[0]

    colA, colB, colC = st.columns(3)
    colA.metric("Late Risk", f"{supplier_row['avg_late_prob']*100:.1f}%")
    colB.metric("Cancel Risk", f"{supplier_row['avg_cancel_prob']*100:.1f}%")
    colC.metric("Profit Risk", f"{supplier_row['avg_profit_risk']*100:.1f}%")

    st.markdown("### Risk Contribution Breakdown")

    risk_df = pd.DataFrame({
        "Component": ["Late Risk", "Cancel Risk", "Profit Risk"],
        "Score": [
            supplier_row["avg_late_prob"],
            supplier_row["avg_cancel_prob"],
            supplier_row["avg_profit_risk"]
        ]
    })

    fig2, ax2 = plt.subplots()
    ax2.barh(risk_df["Component"], risk_df["Score"])
    ax2.set_xlabel("Risk Score")
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # ─────────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────────
    st.subheader("🌡️ Department Risk Heatmap")

    heat_data = dept_summary.set_index("Department Name") * 100
    heat_data = heat_data.T

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        ax=ax3
    )

    st.pyplot(fig3)
    plt.close()

    # ─────────────────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────────────────
    st.markdown("---")

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Risk Report (CSV)",
        csv,
        "supplier_risk_report.csv",
        "text/csv"
    )

else:
    st.title("Supplier Risk Monitor")
    st.warning("Run full ML pipeline first.")