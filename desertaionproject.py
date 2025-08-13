-------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ----------------------
# CONFIG
# ----------------------
NIFTY500_TICKERS = [
    "ABB.NS","ACC.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS",
    "ADANITRANS.NS","ALKEM.NS","AMARAJABAT.NS","AMBER.NS","APOLLOHOSP.NS",
    "APOLLOTYRE.NS","ASHOKLEY.NS","ASIANPAINT.NS","AUROPHARMA.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BALKRISIND.NS","BANDHANBNK.NS",
    "BANKBARODA.NS","BANKINDIA.NS","BATAINDIA.NS","BEL.NS","BERGEPAINT.NS",
    "BHARATFORG.NS","BHARTIARTL.NS","BHEL.NS","BIOCON.NS","BOSCHLTD.NS",
    "BPCL.NS","BRITANNIA.NS","CADILAHC.NS","CANBK.NS","CASTROLIND.NS",
    "CHOLAFIN.NS","CIPLA.NS","COALINDIA.NS","DEEPAKNTR.NS","DIVISLAB.NS",
    "DLF.NS","DRREDDY.NS","EICHERMOT.NS","EQUITAS.NS","ESCORTS.NS",
    "EXIDEIND.NS","FEDERALBNK.NS","GAIL.NS","GLENMARK.NS","GODREJCP.NS",
    "GRASIM.NS","HAVELLS.NS","HCLTECH.NS","HDFC.NS","HDFCAMC.NS",
    "HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDPETRO.NS",
    "HINDUNILVR.NS","INDIGO.NS","INDUSINDBK.NS","INFRATEL.NS","INFY.NS",
    "IOB.NS","IOC.NS","IRCTC.NS","ITC.NS","JINDALSTEL.NS",
    "JSWSTEEL.NS","JUBLFOOD.NS","KOTAKBANK.NS","LTI.NS","LT.NS",
    "LUPIN.NS","M&M.NS","MANAPPURAM.NS","MARICO.NS","MARUTI.NS",
    "MCDOWELL-N.NS","NAUKRI.NS","NESTLEIND.NS","NMDC.NS","NTPC.NS",
    "ONGC.NS","PAGEIND.NS","PETRONET.NS","PIDILITIND.NS","PNB.NS",
    "POWERGRID.NS","RAMCOCEM.NS","RECLTD.NS","RELIANCE.NS","SAIL.NS",
    "SBILIFE.NS","SBIN.NS","SHREECEM.NS","SIEMENS.NS","SRF.NS",
    "SUNPHARMA.NS","SUNTV.NS","TATACHEM.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","UPL.NS",
    "ULTRACEMCO.NS","WIPRO.NS","YESBANK.NS","ZEEL.NS"
]

# ----------------------
# DATA FUNCTIONS
# ----------------------
def download_data_multi(tickers, period="2y", interval="1d"):
    try:
        df = yf.download(tickers, period=period, interval=interval, group_by="ticker", progress=False)
        return df
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def compute_features(df, sma_windows=(20, 50, 200), support_window=30):
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win).mean()
    df["Support"] = df["Close"].rolling(window=support_window).min()
    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)
    return df

def get_latest_features_for_ticker(ticker_df, sma_windows, support_window):
    df = compute_features(ticker_df, sma_windows, support_window)
    latest = df.iloc[-1]
    return {
        "Close": latest["Close"],
        "RSI": latest["RSI"],
        "Support": latest["Support"],
        "SMA20": latest.get("SMA20", np.nan),
        "SMA50": latest.get("SMA50", np.nan),
        "SMA200": latest.get("SMA200", np.nan),
        "Bullish_Div": latest["Bullish_Div"],
        "Bearish_Div": latest["Bearish_Div"],
    }

def get_features_for_all(tickers, sma_windows, support_window):
    multi_df = download_data_multi(tickers)
    if multi_df is None or multi_df.empty:
        return pd.DataFrame()

    features_list = []
    for ticker in tickers:
        try:
            ticker_df = multi_df[ticker].dropna()
            if ticker_df.empty:
                continue
            feats = get_latest_features_for_ticker(ticker_df, sma_windows, support_window)
            feats["Ticker"] = ticker
            features_list.append(feats)
        except Exception:
            continue
    return pd.DataFrame(features_list)

# ----------------------
# STRATEGY
# ----------------------
def predict_buy_sell(df, rsi_buy=30, rsi_sell=70):
    results = df.copy()
    results["Buy_Point"] = (
        (results["RSI"] < rsi_buy) &
        (results["Bullish_Div"]) &
        (np.abs(results["Close"] - results["Support"]) < 0.03 * results["Close"]) &
        (results["Close"] > results["SMA20"]) &
        (results["SMA20"] > results["SMA50"]) &
        (results["SMA50"] > results["SMA200"])
    )
    results["Sell_Point"] = (
        ((results["RSI"] > rsi_sell) & (results["Bearish_Div"])) |
        (results["Close"] < results["Support"]) |
        ((results["SMA20"] < results["SMA50"]) & (results["SMA50"] < results["SMA200"]))
    )
    return results

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title="Nifty500 Interactive Stock Predictor", layout="wide")
st.title("📊 Nifty500 Interactive Stock Buy/Sell Predictor")

with st.sidebar:
    st.header("Settings")
    selected_tickers = st.multiselect("Select stocks", NIFTY500_TICKERS, default=NIFTY500_TICKERS[:10])
    sma_w1 = st.number_input("SMA Window 1", 5, 250, 20)
    sma_w2 = st.number_input("SMA Window 2", 5, 250, 50)
    sma_w3 = st.number_input("SMA Window 3", 5, 250, 200)
    support_window = st.number_input("Support Window (days)", 5, 90, 30)
    rsi_buy = st.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell = st.slider("RSI Sell Threshold", 50, 90, 70)
    run_btn = st.button("Run Analysis")

if run_btn:
    with st.spinner("📥 Fetching data and calculating indicators..."):
        feats = get_features_for_all(selected_tickers, (sma_w1, sma_w2, sma_w3), support_window)
        if feats.empty:
            st.error("⚠ No data available for selected tickers.")
        else:
            preds = predict_buy_sell(feats, rsi_buy, rsi_sell)
            tab1, tab2, tab3 = st.tabs(["✅ Buy Signals", "❌ Sell Signals", "📈 Charts"])

            with tab1:
                st.dataframe(preds[preds["Buy_Point"]])

            with tab2:
                st.dataframe(preds[preds["Sell_Point"]])

            with tab3:
                ticker_chart = st.selectbox("Select Ticker for Chart", selected_tickers)
                chart_df = yf.download(ticker_chart, period="6mo", interval="1d", progress=False)
                if not chart_df.empty:
                    chart_df = compute_features(chart_df, (sma_w1, sma_w2, sma_w3), support_window)
                    st.line_chart(chart_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
                    st.line_chart(chart_df[["RSI"]])

            st.download_button("📥 Download Results", preds.to_csv(index=False).encode(), "nifty500_signals.csv", "text/csv")

st.markdown("⚠ Disclaimer: Educational use only — not financial advice.")
