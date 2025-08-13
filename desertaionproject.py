import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Stock Buy/Sell Predictor", layout="wide")

# ---------------- FEATURE ENGINEERING ----------------
def compute_features(df, sma_windows=(20, 50, 200), support_window=30):
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()

    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)

    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)
    return df.dropna()

# ---------------- RULE-BASED LABEL CREATION ----------------
def label_data_with_rules(df, rsi_buy=30, rsi_sell=70):
    df = df.copy()

    df["Reversal_Buy"] = (
        (df["RSI"] < rsi_buy) &
        (df["Bullish_Div"]) &
        (np.abs(df["Close"] - df["Support"]) < 0.03 * df["Close"]) &
        (df["Close"] > df["SMA20"])
    )

    df["Trend_Buy"] = (
        (df["Close"] > df["SMA20"]) &
        (df["SMA20"] > df["SMA50"]) &
        (df["SMA50"] > df["SMA200"]) &
        (df["RSI"] > 50)
    )

    df["Buy_Point"] = df["Reversal_Buy"] | df["Trend_Buy"]

    df["Sell_Point"] = (
        ((df["RSI"] > rsi_sell) & (df["Bearish_Div"])) |
        (df["Close"] < df["Support"]) |
        ((df["SMA20"] < df["SMA50"]) & (df["SMA50"] < df["SMA200"]))
    )

    df["Signal"] = 0
    df.loc[df["Buy_Point"], "Signal"] = 1
    df.loc[df["Sell_Point"], "Signal"] = -1

    return df

# ---------------- TRAIN ML MODEL ----------------
def train_ml_model(df):
    df = df.dropna(subset=["Signal"])
    features = ["RSI", "SMA20", "SMA50", "SMA200", "Support"]
    X = df[features]
    y = df["Signal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    st.subheader("📊 Model Performance")
    st.text(classification_report(y_test, model.predict(X_test)))

    return model, features

# ---------------- PREDICT WITH MODEL ----------------
def predict_with_model(df, model, features):
    X = df[features]
    df["ML_Pred_Label"] = model.predict(X)
    df["ML_Pred"] = df["ML_Pred_Label"].map({1: "BUY", 0: "HOLD", -1: "SELL"})

    proba = model.predict_proba(X)
    class_order = list(model.classes_)
    df["Prob_Buy"] = proba[:, class_order.index(1)]
    df["Prob_Hold"] = proba[:, class_order.index(0)]
    df["Prob_Sell"] = proba[:, class_order.index(-1)]

    return df

# ---------------- STREAMLIT UI ----------------
st.title("📈 Stock Buy/Sell Predictor (Rules + ML)")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
period = st.selectbox("Select period", ["6mo", "1y", "2y", "5y"])
interval = st.selectbox("Select interval", ["1d", "1wk", "1mo"])

if st.button("Run Analysis"):
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)

    st.write("📄 Raw Data", df.head())

    df_features = compute_features(df)
    df_labeled = label_data_with_rules(df_features)

    model, features = train_ml_model(df_labeled)
    results = predict_with_model(df_labeled, model, features)

    st.subheader("📋 Predictions")
    st.dataframe(results[["Date", "Close", "ML_Pred", "Prob_Buy", "Prob_Hold", "Prob_Sell"]].style.format({
        "Prob_Buy": "{:.2%}",
        "Prob_Hold": "{:.2%}",
        "Prob_Sell": "{:.2%}"
    }))
