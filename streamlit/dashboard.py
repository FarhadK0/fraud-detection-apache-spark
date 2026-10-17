import streamlit as st
import pandas as pd
import plotly.express as px
from textwrap import dedent

#----------------------------------------------
# Page Setup
#----------------------------------------------
st.set_page_config(
    page_title = "Fraud Detection Dashboard",
    page_icon = "🛡️",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#----------------------------------------------
# Theme
#----------------------------------------------
BG = "#0b1220"
PANEL = "#111827"
PANEL_2 = "#162033"
BORDER = "#24324a"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BLUE = "#3b82f6"
RED = "#ef4444"
GREEN = "#22c55e"
ORANGE = "#f59e0b"
PURPLE = "#8b5cf6"
TEAL = "#14b8a6"
GRID = "#26364f"

st.markdown(
    """
    <style>
    .stApp{
    background: linear-gradient(180deg, #0b1220 0%, #0e1728 100%);
    color: #f8fafc;
    }

    .block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    }

    header[data-testid="stHeader"] {
    background: rgba(11, 18, 32, 0.78);
    background-filter: blur(8px);
    }

     section[data-testid="stSidebar"] {
        background-color: #0a1020;
        border-right: 1px solid #24324a;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    h1, h2, h3 {
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: -0.4rem;
        margin-bottom: 1.1rem;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.35rem 0 0.8rem 0;
    }

    .kpi-card {
        background: linear-gradient(145deg, #111827, #162033);
        border: 1px solid #24324a;
        border-radius: 16px;
        padding: 18px;
        min-height: 118px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .kpi-note {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 10px;
    }

    .insight-card {
        background: #111827;
        border: 1px solid #24324a;
        border-left: 4px solid #3b82f6;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #e2e8f0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #24324a;
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #162033);
        border: 1px solid #24324a;
        border-radius: 14px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] { color: #94a3b8; }
    div[data-testid="stMetricValue"] { color: #ffffff; }

    hr {
        border: none;
        border-top: 1px solid #24324a;
        margin: 1rem 0 1.25rem 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding-top: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

def render_html(content):
    st.markdown(dedent(content), unsafe_allow_html=True)

#----------------------------------------------
# Load data
#----------------------------------------------

model_comparison = pd.read_csv("model_comparison.csv")
fraud_distribution = pd.read_csv("fraud_distribution.csv")
product_distribution = pd.read_csv("product_distribution.csv")
card_distribution = pd.read_csv("card_distribution.csv")
device_distribution = pd.read_csv("device_distribution.csv")
feature_importance = pd.read_csv("feature_importance.csv")
transaction_amount = pd.read_csv("transaction_amount.csv")

# ------------------------------------------------------------
# chart styling
# ------------------------------------------------------------

def style_chart(fig, show_legend=True):
    fig.update_layout(paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=12),
    showlegend=show_legend,
    legend_title_text="",
    margin=dict(l=35, r=25, t=35, b=35),
    hoverlabel=dict(bgcolor=PANEL, font_color="#ffffff"),
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(color=TEXT),
        tickfont=dict(color=TEXT),
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        title_font=dict(color=TEXT),
        tickfont=dict(color=TEXT),
    )
    return fig

# ------------------------------------------------------------
# KPI calculations
# ------------------------------------------------------------

fraud_transactions = int(fraud_distribution.loc[fraud_distribution["isFraud"] ==1, "count"].iloc[0]
)

non_fraud_transactions = int(fraud_distribution.loc[fraud_distribution["isFraud"] ==0, "count"].iloc[0]
)

total_transactions = fraud_transactions + non_fraud_transactions
fraud_rate = fraud_transactions / total_transactions * 100

best_model_row = model_comparison.loc[model_comparison["ROC-AUC"].idxmax()]
best_model = str(best_model_row["Model"])
# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.markdown(" ##  🛡️ Fraud Detection")
st.sidebar.caption("Financial Transaction Analytics")

page = st.sidebar.radio("Navigation", ["Overview", "Fraud Analysis", "Model Performance"])

st.sidebar.divider()
st.sidebar.markdown(""" **Final Year Project**

Apache Spark MLlib

Streamlit Dashboard
""")

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("# Fraud Detection in Financial Transactions")
render_html(
    """
    <div class="subtitle">
    Interactive dashboard presenting fraud patterns, transaction behaviour, and machine-learning model performance.
    </div>
    """
)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    render_html('<div class="section-title">Key Performance Indicators </div>')

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        render_html(
            f"""
            <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value" >{total_transactions:,}</div>
            <div class="kpi-note">Transactions analysed</div>
            </div>
            """
        )

    with c2:
        render_html(
            f"""
            <div class="kpi-card"> 
                <div class="kpi-label"> Fraud Transactions</div>
                <div class="kpi-value" style="color:{RED};"> {fraud_transactions:,}</div>
                <div class="kpi-note">Fraudulent transactions</div>
            </div>
            """
        )

    with c3:
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Fraud Rate</div>
                <div class="kpi-value" style="color:{ORANGE};">{fraud_rate:.2f}%</div>
                <div class="kpi-note">Percentage of fraud</div>
            </div>
            """
        )

    with c4:
        render_html(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Best Model</div>
                <div class="kpi-value" style="color:{GREEN}; font-size:1.45rem;">{best_model}</div>
                <div class="kpi-note">Strongest overall performance</div>
            </div>
            """
        )

    st.divider()
    left, right = st.columns(2)
    with left:
     render_html('<div class="section-title">Fraud vs Non-Fraud Transactions</div>')

     fraud_chart_data = fraud_distribution.copy()
     fraud_chart_data["Class"] = fraud_chart_data["isFraud"].map({0: "Non-Fraud", 1: "Fraud"})

     fraud_fig = px.pie(
          fraud_chart_data,
          names="Class",
          values="count",
          hole=0.64,
          color="Class",
          color_discrete_map={"Non-Fraud": BLUE, "Fraud": RED},
     )

     fraud_fig.update_traces(
          textinfo="percent+label",
          textfont_color="#ffffff")
     
     fraud_fig.update_layout(
          annotations=[
               dict(
                    text=f"<b>{total_transactions:,}</b><br>Total",
                    x=0.5,
                    y=0.5,
                    font=dict(size=18, color="#ffffff"),
                    showarrow=False
               )
          ]
     )

     style_chart(fraud_fig)
     st.plotly_chart(fraud_fig, use_container_width=True)

    with right:
        render_html('<div class="section-title">Product Category Distribution</div>')

        product_sorted = product_distribution.sort_values("count", ascending=False).copy()

        # ProductCD values are anonymised in the IEEE-CIS dataset,
        # so clearer labels are used for dashboard readability.
        product_sorted["Product Category"] = (
            "Category " + product_sorted["ProductCD"].astype(str)
        )

        product_fig = px.bar(
            product_sorted,
            x="Product Category",
            y="count",
            text="count",
            color_discrete_sequence=[BLUE],
        )
        product_fig.update_traces(textposition="outside")
        product_fig.update_layout(
            xaxis_title="Anonymised Product Category",
            yaxis_title="Number of Transactions",
        )
        style_chart(product_fig, show_legend=False)
        st.plotly_chart(product_fig, use_container_width=True)

        st.caption(
            "ProductCD contains anonymised product categories. "
            "Categories W, C, R, H and S are coded labels, and the dataset "
            "does not provide their real-world product names."
        )

    left, right = st.columns(2)

    with left:
     render_html('<div class="section-title">Card Network Distribution</div>')

     card_clean = card_distribution.copy()
     card_clean["card4"] = card_clean["card4"].fillna("Unknown")
     card_clean = card_clean.sort_values("count", ascending=False)

     card_fig = px.bar(
            card_clean,
            x="card4",
            y="count",
            text="count",
            color_discrete_sequence=[ORANGE],
     )

     card_fig.update_traces(textposition="outside")
     card_fig.update_layout(
            xaxis_title="Card Network",
            yaxis_title="Number of Transactions",
     )
     style_chart(card_fig, show_legend=False)
     st.plotly_chart(card_fig, use_container_width=True)

    with right:
        render_html('<div class="section-title">Device Type Distribution</div>')

        device_fig = px.pie(
            device_distribution,
            names="DeviceType",
            values="count",
            hole=0.58,
            color="DeviceType",
            color_discrete_map={
                "desktop": BLUE,
                "mobile": PURPLE,
                "Desktop": BLUE,
                "Mobile": PURPLE,
            },
        )
        device_fig.update_traces(textinfo="percent+label", textfont_color="#ffffff")
        style_chart(device_fig)
        st.plotly_chart(device_fig, use_container_width=True)

    render_html('<div class="section-title">Distribution of Transaction Amounts</div>')

    if "TransactionAmt" in transaction_amount.columns:
        amount_column = "TransactionAmt"
    elif "TransactionAmount" in transaction_amount.columns:
        amount_column = "TransactionAmount"
    else:
        amount_column = transaction_amount.columns[0]

    amount_limit = transaction_amount[amount_column].quantile(0.99)
    transaction_filtered = transaction_amount[
        transaction_amount[amount_column] <= amount_limit
    ]

    amount_fig = px.histogram(
        transaction_filtered,
        x=amount_column,
        nbins=50,
        color_discrete_sequence=[TEAL],
    )
    amount_fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Number of Transactions",
        bargap=0.05,
    )
    style_chart(amount_fig, show_legend=False)
    st.plotly_chart(amount_fig, use_container_width=True)

    st.caption(
        "Each bar represents the number of transactions within a specific "
        "transaction amount range. Most transactions are concentrated at "
        "lower amounts, while higher-value transactions occur less frequently."
    )

# ============================================================
# FRAUD ANALYSIS
# ============================================================
elif page == "Fraud Analysis":

    st.markdown("## Fraud Analysis")
    st.caption("Distribution analysis of important transaction characteristics.")

    col1, col2 = st.columns(2)

    with col1:
        render_html('<div class="section-title">Product Category Share</div>')

        product_pct = product_distribution.sort_values("Percentage", ascending=False)
        product_fig = px.bar(
            product_pct,
            x="ProductCD",
            y="Percentage",
            text="Percentage",
            color_discrete_sequence=[RED],
        )
        product_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )
        product_fig.update_layout(
            xaxis_title="Product Category",
            yaxis_title="Percentage (%)",
        )
        style_chart(product_fig, show_legend=False)
        st.plotly_chart(product_fig, use_container_width=True)

    with col2:
        render_html('<div class="section-title">Card Network Share</div>')

        card_pct = card_distribution.copy()
        card_pct["card4"] = card_pct["card4"].fillna("Unknown")
        card_pct = card_pct.sort_values("Percentage", ascending=False)

        card_fig = px.bar(
            card_pct,
            x="card4",
            y="Percentage",
            text="Percentage",
            color_discrete_sequence=[ORANGE],
        )
        card_fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )
        card_fig.update_layout(
            xaxis_title="Card Network",
            yaxis_title="Percentage (%)",
        )
        style_chart(card_fig, show_legend=False)
        st.plotly_chart(card_fig, use_container_width=True)

    render_html('<div class="section-title">Device Type Share</div>')

    device_pct = device_distribution.sort_values("Percentage", ascending=False)
    device_fig = px.bar(
        device_pct,
        x="DeviceType",
        y="Percentage",
        text="Percentage",
        color="DeviceType",
        color_discrete_map={
            "desktop": BLUE,
            "mobile": PURPLE,
            "Desktop": BLUE,
            "Mobile": PURPLE,
        },
    )
    device_fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )
    device_fig.update_layout(
        xaxis_title="Device Type",
        yaxis_title="Percentage (%)",
    )
    style_chart(device_fig)
    st.plotly_chart(device_fig, use_container_width=True)

    render_html('<div class="section-title">Key Insights</div>')

    render_html(
        f"""
        <div class="insight-card">
            Fraud transactions represent <b>{fraud_rate:.2f}%</b> of the analysed transactions.
        </div>

        <div class="insight-card" style="border-left-color:{ORANGE};">
            Product and card distributions are highly uneven, showing that some categories dominate transaction activity.
        </div>

        <div class="insight-card" style="border-left-color:{PURPLE};">
            Desktop and mobile transactions show different usage patterns in the dataset.
        </div>
        """
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.markdown("## Machine Learning Model Performance")
    st.caption(
        "Comparison of Logistic Regression, Decision Tree, and Random Forest. "
        "The KPI cards below show the performance of the selected Random Forest model."
    )

    model_display = model_comparison.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for column in metric_columns:
        if model_display[column].max() <= 1:
            model_display[column] = model_display[column] * 100
        model_display[column] = model_display[column].round(2)

    rf_rows = model_display[
        model_display["Model"].astype(str).str.lower().str.contains("random forest")
    ]

    if not rf_rows.empty:
        rf_row = rf_rows.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Random Forest Accuracy", f"{rf_row['Accuracy']:.2f}%")
        m2.metric("Random Forest Weighted Precision", f"{rf_row['Precision']:.2f}%")
        m3.metric("Random Forest Weighted Recall", f"{rf_row['Recall']:.2f}%")
        m4.metric("Random Forest F1-Score", f"{rf_row['F1-Score']:.2f}%")
        m5.metric("Random Forest ROC-AUC", f"{rf_row['ROC-AUC']:.2f}%")

    st.markdown("---")

    render_html('<div class="section-title">Model Comparison Table</div>')
    st.dataframe(model_display, use_container_width=True, hide_index=True)

    render_html('<div class="section-title">Model Performance Comparison</div>')

    model_long = model_display.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    model_fig = px.bar(
        model_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_map={
            "Accuracy": BLUE,
            "Precision": GREEN,
            "Recall": ORANGE,
            "F1-Score": PURPLE,
            "ROC-AUC": RED,
        },
    )
    model_fig.update_layout(
        xaxis_title="Machine Learning Model",
        yaxis_title="Score (%)",
    )
    model_fig.update_yaxes(range=[0, 100])
    style_chart(model_fig)
    st.plotly_chart(model_fig, use_container_width=True)

    render_html('<div class="section-title">Top 10 Random Forest Feature Importances</div>')

    top_features = (
        feature_importance
        .sort_values("Importance", ascending=False)
        .head(10)
        .sort_values("Importance", ascending=True)
    )

    feature_fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color_discrete_sequence=[TEAL],
    )
    feature_fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )
    feature_fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
    )
    style_chart(feature_fig, show_legend=False)
    st.plotly_chart(feature_fig, use_container_width=True)

    render_html('<div class="section-title">Best Performing Model</div>')

    render_html(
        f"""
        <div class="insight-card" style="border-left-color:{GREEN};">
            <b>{best_model}</b> achieved the strongest overall performance and was selected as the final model for this fraud detection project.
        </div>
        """
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
render_html(
    """
    <div class="footer">
        Fraud Detection in Financial Transactions • Final Year Project • Apache Spark MLlib + Streamlit
    </div>
    """
)