import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === CONFIG ===
st.set_page_config(layout="wide")

MODEL_DIR = "models"
REFRESH_INTERVAL_MS = 5000  # 5 seconds refresh
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.01  # 1% of balance

# === SIDEBAR CONFIGURATION ===
st.sidebar.title("⚙️ Settings")
LOOKBACK = st.sidebar.slider("Lookback Window", 10, 100, 20)
TP = st.sidebar.slider("Take Profit (points)", 100, 1000, 500)
SL = st.sidebar.slider("Stop Loss (points)", 100, 1000, 500)
FAST_FORWARD = st.sidebar.checkbox("⚡ Fast Forward Mode")

st.sidebar.markdown("### Position Sizing")
USE_FIXED_LOT = st.sidebar.checkbox("Use Fixed Lot Size Instead of Risk-Based", False)
FIXED_LOT_SIZE = st.sidebar.number_input("Fixed Lot Size", min_value=0.01, value=1.0, step=0.01)

# === AUTO-REFRESH ===
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="datarefresh") if not FAST_FORWARD else 0

# === TITLE ===
st.title("📈 XAUUSD Live Simulated Streaming + Backtest")

# === CSV UPLOADER ===
uploaded_file = st.file_uploader("📤 Upload XAUUSD CSV File", type=["csv"])
if uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)
    df_full['time'] = pd.to_datetime(df_full['time'])
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# === SELECT MODEL & SCALER ===
model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("model_")])
scaler_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("scaler_")])

selected_model_file = st.selectbox("🧠 Select Model", model_files)
selected_scaler_file = st.selectbox("🔧 Select Scaler", scaler_files)

model = joblib.load(os.path.join(MODEL_DIR, selected_model_file))
scaler = joblib.load(os.path.join(MODEL_DIR, selected_scaler_file))

st.success("✅ Model and scaler loaded successfully.")

# === SIMULATE LIVE DATA ===
if len(df_full) < LOOKBACK + 10:
    st.error("Not enough data in CSV for simulation.")
    st.stop()

df = df_full.copy() if FAST_FORWARD else df_full.iloc[:min(LOOKBACK + count, len(df_full))].copy()

# === HELPER FUNCTIONS ===
def is_ranging(data, threshold=0.001):
    closes = data['close'].values
    max_close = np.max(closes[-LOOKBACK:])
    min_close = np.min(closes[-LOOKBACK:])
    return (max_close - min_close) / min_close < threshold

def extract_features(df, i):
    closes = df['close'].values
    ma = pd.Series(closes).rolling(window=LOOKBACK).mean().values
    returns = np.diff(closes) / closes[:-1]
    close_features = closes[i - LOOKBACK:i]
    return_features = returns[i - LOOKBACK:i - 1]
    ma_diff = closes[i] - ma[i]
    features = np.concatenate([close_features, return_features, [ma_diff]])
    return scaler.transform([features])

def position_size(balance, sl_points):
    if USE_FIXED_LOT:
        return FIXED_LOT_SIZE
    risk_dollars = balance * RISK_PER_TRADE
    return risk_dollars / (sl_points * 0.01)

def run_backtest(df, model, scaler):
    balance = INITIAL_BALANCE
    trade_log, trade_markers, entry_markers = [], [], []
    open_trade = None
    y_true, y_pred = [], []

    for i in range(LOOKBACK, len(df) - 1):
        if open_trade:
            trade_type, entry_price, entry_time, lot_size = open_trade
            high = df['high'].iloc[i + 1]
            low = df['low'].iloc[i + 1]
            time_stamp = df['time'].iloc[i + 1]
            if trade_type == "BUY":
                sl = entry_price - SL * 0.01
                tp = entry_price + TP * 0.01
                if high >= tp:
                    pnl = TP * 0.01 * lot_size
                    balance += pnl
                    trade_log.append((time_stamp, "BUY TP", pnl, balance))
                    trade_markers.append((time_stamp, tp, "BUY TP"))
                    open_trade = None
                elif low <= sl:
                    pnl = -SL * 0.01 * lot_size
                    balance += pnl
                    trade_log.append((time_stamp, "BUY SL", pnl, balance))
                    trade_markers.append((time_stamp, sl, "BUY SL"))
                    open_trade = None
            elif trade_type == "SELL":
                sl = entry_price + SL * 0.01
                tp = entry_price - TP * 0.01
                if low <= tp:
                    pnl = TP * 0.01 * lot_size
                    balance += pnl
                    trade_log.append((time_stamp, "SELL TP", pnl, balance))
                    trade_markers.append((time_stamp, tp, "SELL TP"))
                    open_trade = None
                elif high >= sl:
                    pnl = -SL * 0.01 * lot_size
                    balance += pnl
                    trade_log.append((time_stamp, "SELL SL", pnl, balance))
                    trade_markers.append((time_stamp, sl, "SELL SL"))
                    open_trade = None
            continue

        try:
            features = extract_features(df, i)
        except Exception:
            continue

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]
        confidence = prob[prediction]

        y_pred.append(prediction)
        y_true.append(1 if df['close'].iloc[i + 1] > df['close'].iloc[i] else 0)

        if confidence < 0.7 or is_ranging(df.iloc[i - LOOKBACK:i]):
            continue

        entry_price = df['close'].iloc[i]
        time_stamp = df['time'].iloc[i + 1]
        lot_size = position_size(balance, SL)

        if prediction == 1:
            open_trade = ("BUY", entry_price, time_stamp, lot_size)
            entry_markers.append((time_stamp, entry_price, "BUY"))
        else:
            open_trade = ("SELL", entry_price, time_stamp, lot_size)
            entry_markers.append((time_stamp, entry_price, "SELL"))

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

    return trade_log, balance, trade_markers, entry_markers, metrics

def plot_candles(df, trade_markers=[], entry_markers=[]):
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        increasing_line_color='green', decreasing_line_color='red',
        name="Price"
    )])
    for t in trade_markers:
        fig.add_trace(go.Scatter(
            x=[t[0]], y=[t[1]], mode="markers+text", text=[t[2]],
            textposition="top center", marker=dict(size=10, color="blue" if "BUY" in t[2] else "red"),
            name=t[2]
        ))
    for t in entry_markers:
        fig.add_trace(go.Scatter(
            x=[t[0]], y=[t[1]], mode="markers",
            marker=dict(size=8, color="black", symbol="triangle-up" if t[2] == "BUY" else "triangle-down"),
            name=f"Entry {t[2]}"
        ))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    return fig

# === RUN BACKTEST ===
trade_log, final_balance, trade_markers, entry_markers, metrics = run_backtest(df, model, scaler)

# === TRADE LOG ===
st.subheader("📜 Trade Log")
trade_df = pd.DataFrame(trade_log, columns=["Time", "Action", "PnL", "Balance"])
st.dataframe(trade_df)

csv = trade_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Trade Log", csv, "trade_log.csv", "text/csv")

st.success(f"💰 Balance after Backtest: ${final_balance:.2f}")

# === METRICS ===
st.subheader("📊 Performance Metrics")
for metric, value in metrics.items():
    st.metric(metric, f"{value:.2%}")

# === EQUITY CURVE ===
st.subheader("📈 Equity Curve")
if not trade_df.empty:
    fig_eq, ax = plt.subplots()
    ax.plot(trade_df["Time"], trade_df["Balance"], color="green")
    ax.set_xlabel("Time")
    ax.set_ylabel("Balance")
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig_eq)
else:
    st.write("No trades executed yet.")

# === CANDLE CHART ===
st.subheader("📊 Live Candlestick Chart")
candles_df = df.tail(LOOKBACK + 10)
st.plotly_chart(plot_candles(candles_df, trade_markers, entry_markers), use_container_width=True)

# === LIVE PREDICTION ===
st.subheader("🔮 Live Prediction on Latest Candle")
if len(df) > LOOKBACK:
    i = len(df) - 1
    try:
        live_features = extract_features(df, i)
        prediction = model.predict(live_features)[0]
        prob = model.predict_proba(live_features)[0]
        if prediction == 1:
            st.markdown(
                f"<div style='padding:10px;background-color:#d4edda;border-left:5px solid #28a745;'>"
                f"<b>Prediction:</b> <span style='color:#28a745;'>BUY</span><br>"
                f"<b>Confidence:</b> {prob[1]:.2%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='padding:10px;background-color:#f8d7da;border-left:5px solid #dc3545;'>"
                f"<b>Prediction:</b> <span style='color:#dc3545;'>SELL</span><br>"
                f"<b>Confidence:</b> {prob[0]:.2%}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Error with live prediction: {e}")
else:
    st.warning("⚠️ Not enough data for live prediction.")
