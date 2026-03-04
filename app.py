"""
app.py — Zero-Waste Supermarket Dashboard
==========================================
Premium website-style Streamlit UI tying together the DQN
pricing agent and GenAI marketing nudge generator.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from rl_env import DISCOUNT_MAP
from train_agent import predict_discount, MODEL_PATH
from genai_nudge import generate_nudge


# ──────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zero-Waste Supermarket",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────
# Global CSS — Premium Website Theme
# ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f0fdf4 0%, #f8fafc 40%, #ffffff 100%);
    }
    .block-container {
        padding-top: 1rem !important;
        max-width: 1200px;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ── Navbar ── */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .navbar-brand .logo {
        font-size: 1.8rem;
    }
    .navbar-brand .name {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
    }
    .navbar-brand .tag {
        font-size: 0.65rem;
        font-weight: 600;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .navbar-links {
        display: flex;
        gap: 1.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
    }
    .navbar-links span {
        cursor: default;
        transition: color 0.2s;
    }

    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 3rem 1rem 2.5rem;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        color: #0f172a;
        margin: 0 0 0.8rem;
        letter-spacing: -1.5px;
        line-height: 1.1;
    }
    .hero h1 .accent {
        background: linear-gradient(135deg, #10b981, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        font-size: 1.15rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto 1.8rem;
        line-height: 1.6;
    }

    /* ── Upload Area ── */
    .upload-zone {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        max-width: 550px;
        margin: 0 auto 2rem;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #10b981;
        box-shadow: 0 0 0 4px rgba(16,185,129,0.1);
    }
    .upload-zone .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .upload-zone .text {
        font-size: 0.95rem;
        color: #64748b;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f1f5f9;
    }
    .section-header .icon { font-size: 1.3rem; }
    .section-header h2 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }

    /* ── Metric Cards ── */
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    @media (max-width: 768px) {
        .cards-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.25s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }
    .card .card-icon {
        width: 48px; height: 48px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.8rem;
        font-size: 1.3rem;
    }
    .card .card-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    .card .card-value {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .card .card-sub {
        font-size: 0.78rem;
        margin-top: 0.15rem;
        font-weight: 500;
    }

    /* Card color themes */
    .card-green .card-icon { background: #ecfdf5; }
    .card-green .card-label { color: #059669; }
    .card-green .card-value { color: #065f46; }
    .card-green .card-sub { color: #10b981; }

    .card-amber .card-icon { background: #fffbeb; }
    .card-amber .card-label { color: #d97706; }
    .card-amber .card-value { color: #92400e; }
    .card-amber .card-sub { color: #f59e0b; }

    .card-blue .card-icon { background: #eff6ff; }
    .card-blue .card-label { color: #2563eb; }
    .card-blue .card-value { color: #1e3a8a; }
    .card-blue .card-sub { color: #3b82f6; }

    .card-rose .card-icon { background: #fff1f2; }
    .card-rose .card-label { color: #e11d48; }
    .card-rose .card-value { color: #9f1239; }
    .card-rose .card-sub { color: #fb7185; }

    /* ── Nudge Notification ── */
    .nudge-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        max-width: 500px;
        margin: 1rem auto;
    }
    .nudge-top {
        background: linear-gradient(135deg, #10b981, #059669);
        padding: 0.8rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .nudge-top .bell { font-size: 1.1rem; }
    .nudge-top .title {
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .nudge-top .time {
        margin-left: auto;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.8);
    }
    .nudge-body {
        padding: 1.3rem 1.5rem;
        font-size: 1rem;
        line-height: 1.65;
        color: #1e293b;
    }

    /* ── Feature Cards (landing) ── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin: 1.5rem 0 3rem;
    }
    @media (max-width: 768px) {
        .features-grid { grid-template-columns: 1fr; }
    }
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        border-color: #10b981;
    }
    .feature-card .f-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
    }
    .feature-card h3 {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.5rem;
    }
    .feature-card p {
        font-size: 0.88rem;
        color: #64748b;
        line-height: 1.5;
        margin: 0;
    }

    /* ── Data Table Styling ── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu, footer { visibility: hidden; }
    div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# Navbar
# ──────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span class="logo">🥦</span>
        <span class="name">ZeroWaste</span>
        <span class="tag">AI-Powered</span>
    </div>
    <div class="navbar-links">
        <span>Dashboard</span>
        <span>Analytics</span>
        <span>Settings</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# File Upload (hidden in a clean way)
# ──────────────────────────────────────────────────────
uploaded_file = None

if "df" not in st.session_state:
    # ── Hero / Landing Page ──
    st.markdown("""
    <div class="hero">
        <h1>Reduce Food Waste with<br><span class="accent">Intelligent AI Pricing</span></h1>
        <p>
            Our hybrid AI system uses Reinforcement Learning and Generative AI
            to dynamically optimize discounts and craft compelling marketing — 
            before your perishables expire.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-zone"><div class="icon">📂</div><div class="text">Upload your inventory CSV to get started</div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload perishable_goods_management.csv",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ["product_name", "base_price", "days_until_expiry",
                             "initial_quantity", "daily_demand"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                st.stop()
            st.session_state.df = df
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            st.stop()

    # ── Feature Cards ──
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="f-icon">🧠</div>
            <h3>Smart Pricing Engine</h3>
            <p>Deep Q-Network agent trained on your inventory data finds the optimal discount to maximize revenue while clearing expiring stock.</p>
        </div>
        <div class="feature-card">
            <div class="f-icon">📱</div>
            <h3>AI Marketing Copy</h3>
            <p>GPT-powered nudge generator creates psychologically-framed push notifications that drive customer purchases — no copywriter needed.</p>
        </div>
        <div class="feature-card">
            <div class="f-icon">♻️</div>
            <h3>Zero Waste Mission</h3>
            <p>Data-driven inventory management reduces food waste by matching dynamic pricing with real-time demand patterns.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">Zero-Waste Supermarket &nbsp;·&nbsp; Powered by DQN + GPT &nbsp;·&nbsp; Built with Streamlit</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════
# DASHBOARD VIEW (after CSV upload)
# ══════════════════════════════════════════════════════
df = st.session_state.df

# ── Quick Stats Bar ──
total_items = len(df)
at_risk_count = len(df[df["days_until_expiry"] <= 5])
avg_price = df["base_price"].mean()
total_stock = df["initial_quantity"].sum()

st.markdown(f"""
<div class="cards-grid">
    <div class="card card-green">
        <div class="card-icon card-green">📦</div>
        <div class="card-label">Total Items</div>
        <div class="card-value">{total_items:,}</div>
        <div class="card-sub">in inventory</div>
    </div>
    <div class="card card-rose">
        <div class="card-icon card-rose">🚨</div>
        <div class="card-label">At-Risk</div>
        <div class="card-value">{at_risk_count:,}</div>
        <div class="card-sub">expiring ≤ 5 days</div>
    </div>
    <div class="card card-blue">
        <div class="card-icon card-blue">💰</div>
        <div class="card-label">Avg Price</div>
        <div class="card-value">${avg_price:.2f}</div>
        <div class="card-sub">across catalog</div>
    </div>
    <div class="card card-amber">
        <div class="card-icon card-amber">📊</div>
        <div class="card-label">Total Stock</div>
        <div class="card-value">{total_stock:,}</div>
        <div class="card-sub">units on shelf</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Inventory Table ──
st.markdown("""
<div class="section-header">
    <span class="icon">📋</span>
    <h2>Inventory Overview</h2>
</div>
""", unsafe_allow_html=True)

st.dataframe(df.head(50), use_container_width=True, height=400)


# ── At-Risk Product Selector ──
st.markdown("""
<div class="section-header">
    <span class="icon">⚡</span>
    <h2>AI Discount Optimizer</h2>
</div>
""", unsafe_allow_html=True)

at_risk = df[df["days_until_expiry"] <= 5].copy()

if at_risk.empty:
    st.warning("No items found with `days_until_expiry ≤ 5` in the uploaded data.")
    st.stop()

at_risk["display_label"] = (
    at_risk["product_name"] + "  ·  $"
    + at_risk["base_price"].round(2).astype(str)
    + "  ·  " + at_risk["days_until_expiry"].astype(str) + "d left"
)

col_select, col_spacer = st.columns([2, 1])
with col_select:
    selected_label = st.selectbox(
        "Select an at-risk product to optimize:",
        options=at_risk["display_label"].unique().tolist(),
        index=0,
    )

selected_row = at_risk[at_risk["display_label"] == selected_label].iloc[0]
days_left = int(selected_row["days_until_expiry"])
stock = int(selected_row["initial_quantity"])
demand = float(selected_row["daily_demand"])
base_price = float(selected_row["base_price"])
product_name = str(selected_row["product_name"])


# ── Model Prediction ──
model_zip = MODEL_PATH + ".zip"
if not os.path.exists(model_zip):
    st.error("⚠️ Trained model not found! Run `python train_agent.py` first.")
    st.stop()

max_days = max(float(df["days_until_expiry"].max()), 1.0)
max_stock = max(float(df["initial_quantity"].max()), 1.0)
max_demand = max(float(df["daily_demand"].max()), 1.0)

state = np.array([
    days_left / max_days,
    stock / max_stock,
    demand / max_demand,
], dtype=np.float32)

try:
    action_idx, discount = predict_discount(state)
except Exception as e:
    st.error(f"Model prediction error: {e}")
    st.stop()

new_price = base_price * (1.0 - discount)
savings = base_price - new_price
discount_display = int(discount * 100)


# ── Results Cards ──
st.markdown(f"""
<div class="cards-grid">
    <div class="card card-green">
        <div class="card-icon card-green">🏷️</div>
        <div class="card-label">Original Price</div>
        <div class="card-value">${base_price:.2f}</div>
        <div class="card-sub">{product_name}</div>
    </div>
    <div class="card card-amber">
        <div class="card-icon card-amber">🤖</div>
        <div class="card-label">AI Discount</div>
        <div class="card-value">{discount_display}% OFF</div>
        <div class="card-sub">Save ${savings:.2f}</div>
    </div>
    <div class="card card-blue">
        <div class="card-icon card-blue">✨</div>
        <div class="card-label">Optimized Price</div>
        <div class="card-value">${new_price:.2f}</div>
        <div class="card-sub">DQN recommended</div>
    </div>
    <div class="card card-rose">
        <div class="card-icon card-rose">⏰</div>
        <div class="card-label">Expires In</div>
        <div class="card-value">{days_left}d</div>
        <div class="card-sub">{stock:,} units in stock</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── GenAI Marketing Nudge ──
st.markdown("""
<div class="section-header">
    <span class="icon">📱</span>
    <h2>AI Marketing Nudge</h2>
</div>
""", unsafe_allow_html=True)

with st.spinner("✨ GPT is crafting your marketing copy..."):
    nudge_text = generate_nudge(product_name, base_price, discount, days_left)

st.markdown(f"""
<div class="nudge-card">
    <div class="nudge-top">
        <span class="bell">🔔</span>
        <span class="title">Push Notification Preview</span>
        <span class="time">Just now</span>
    </div>
    <div class="nudge-body">
        {nudge_text}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.success("✅ AI optimization complete — discount and marketing nudge ready for deployment!")

# ── Reset Button ──
st.markdown("")
if st.button("🔄 Upload New Dataset", use_container_width=True):
    del st.session_state.df
    st.rerun()

# ── Footer ──
st.markdown('<div class="footer">Zero-Waste Supermarket &nbsp;·&nbsp; Powered by DQN + GPT &nbsp;·&nbsp; Built with Streamlit</div>', unsafe_allow_html=True)
