import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ----------- CONFIG -----------
NIFTY500_TICKERS = [
    "ABB.NS","ACC.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS",
    "ADANITRANS.NS","ALKEM.NS","AMARAJABAT.NS","AMBER.NS","APOLLOHOSP.NS",
    "APOLLOTYRE.NS","ASHOKLEY.NS","ASIANPAINT.NS","AUROPHARMA.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BALKRISIND.NS","BANDHANBNK.NS",
    "BANKBARODA.NS","BANKINDIA.NS","BATAINDIA.NS","BEL.NS","BERGEPAINT.NS",
    "BHARATFORG.NS","BHARTIARTL.NS","BHEL.NS","BIOCON.NS","BOSCHLTD.NS",
    "BPCL.NS","BRITANNIA.NS","CADILAHC.NS","CANBK.NS","CASTROLIND.NS",
    "CHOLAFIN.NS","CIPLA.NS","COALINDIA.NS","DEEPAKNTR.NS","DIVISLAB.NS",
    "DLF.NS","DRREDDY.NS","EICHERMOT.NS","EQUITAS.NS","ESCORTS.NS","EXIDEIND.NS",
    "FEDERALBNK.NS","GAIL.NS","GLENMARK.NS","GODREJCP.NS","GRASIM.NS","HAVELLS.NS",
    "HCLTECH.NS","HDFC.NS","HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS",
    "HINDALCO.NS","HINDPETRO.NS","HINDUNILVR.NS","INDIGO.NS","INDUSINDBK.NS",
    "INFRATEL.NS","INFY.NS","IOB.NS","IOC.NS","IRCTC.NS","ITC.NS","JINDALSTEL.NS",
    "JSWSTEEL.NS","JUBLFOOD.NS","KOTAKBANK.NS","LTI.NS","LT.NS","LUPIN.NS","M&M.NS",
    "MANAPPURAM.NS","MARICO.NS","MARUTI.NS","MCDOWELL-N.NS","NAUKRI.NS","NESTLEIND.NS",
    "NMDC.NS","NTPC.NS","ONGC.NS","PAGEIND.NS","PETRONET.NS","PIDILITIND.NS",
    "PNB.NS","POWERGRID.NS","RAMCOCEM.NS","RECLTD.NS","RELIANCE.NS","SAIL.NS",
    "SBILIFE.NS","SBIN.NS","SHREECEM.NS","SIEMENS.NS","SRF.NS","SUNPHARMA.NS",
    "SUNTV.NS","TATACHEM.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TCS.NS","TECHM.NS","TITAN.NS","UPL.NS","ULTRACEMCO.NS","WIPRO.NS","YESBANK.NS","ZEEL.NS"
]

# ----------- HELPERS -----------
@st.cache_data(show_spinner=False)
def download_data_multi(tickers, period="2y", interval="1d"):
    try:
        return yf.download(
            tickers, period=period, interval=interval,
            group_by="ticker", progress=False
        )
    except Exception:
        return None

def compute_features(df, sma_windows=(20, 50, 200), support_window=30):
    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Ensure 'Close' column exists and is valid
    if "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.DataFrame()

    df = df.copy()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # SMA calculations
    for win in sma_windows:
        df[f"SMA{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()

    # Support level
    df["Support"] = df["Close"].rolling(window=support_window, min_periods=1).min()

    # Divergence features
    df["RSI_Direction"] = df["RSI"].diff(5)
    df["Price_Direction"] = df["Close"].diff(5)
    df["Bullish_Div"] = (df["RSI_Direction"] > 0) & (df["Price_Direction"] < 0)
    df["Bearish_Div"] = (df["RSI_Direction"] < 0) & (df["Price_Direction"] > 0)

    return df

def get_latest_features_for_ticker(ticker_df, ticker, sma_windows, support_window):
    df = compute_features(ticker_df, sma_windows, support_window).dropna()
    if df.empty:
        return None
    latest = df.iloc[-1]
    return {
        "Ticker": ticker,
        "Close": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "Support": float(latest["Support"]),
        **{f"SMA{w}": float(latest.get(f"SMA{w}", np.nan)) for w in sma_windows},
        "Bullish_Div": bool(latest["Bullish_Div"]),
        "Bearish_Div": bool(latest["Bearish_Div"])
    }

def get_features_for_all(tickers, sma_windows, support_window):
    multi_df = download_data_multi(tickers)
    if multi_df is None or multi_df.empty:
        return pd.DataFrame()
    features_list = []
    if isinstance(multi_df.columns, pd.MultiIndex):
        available = multi_df.columns.get_level_values(0).unique()
        for ticker in tickers:
            if ticker not in available:
                continue
            tdf = multi_df[ticker].dropna()
            if tdf.empty:
                continue
            feats = get_latest_features_for_ticker(tdf, ticker, sma_windows, support_window)
            if feats:
                features_list.append(feats)
    else:
        feats = get_latest_features_for_ticker(multi_df.dropna(), tickers[0], sma_windows, support_window)
        if feats:
            features_list.append(feats)
    return pd.DataFrame(features_list)

# ----------- STRATEGY -----------
def predict_buy_sell(df, rsi_buy=30, rsi_sell=70):
    if df.empty:
        return df
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

# ----------- UI -----------
st.set_page_config(page_title="Nifty500 Buy/Sell Predictor", layout="wide")
st.title("📊 Nifty500 Buy/Sell Predictor")

with st.sidebar:
    st.header("Settings")
    selected_tickers = st.multiselect(
        "Select stocks", NIFTY500_TICKERS, default=NIFTY500_TICKERS[:5]
    )
    sma_w1 = st.number_input("SMA Window 1", 5, 250, 20)
    sma_w2 = st.number_input("SMA Window 2", 5, 250, 50)
    sma_w3 = st.number_input("SMA Window 3", 5, 250, 200)
    support_window = st.number_input("Support Period (days)", 5, 90, 30)
    rsi_buy = st.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell = st.slider("RSI Sell Threshold", 50, 90, 70)
    run_analysis = st.button("Run Analysis")

if run_analysis:
    with st.spinner("Fetching and computing..."):
        feats = get_features_for_all(
            selected_tickers, (sma_w1, sma_w2, sma_w3), support_window
        )
        if feats.empty:
            st.error("No valid data for selected tickers.")
        else:
            preds = predict_buy_sell(feats, rsi_buy, rsi_sell)
            tab1, tab2, tab3 = st.tabs(["Buy Signals", "Sell Signals", "Charts"])

            with tab1:
                st.dataframe(preds[preds["Buy_Point"]])
            with tab2:
                st.dataframe(preds[preds["Sell_Point"]])
            with tab3:
                ticker_for_chart = st.selectbox("Chart Ticker", selected_tickers)
                chart_df = yf.download(
                    ticker_for_chart, period="6mo", interval="1d", progress=False
                )
                if not chart_df.empty:
                    chart_df = compute_features(chart_df, (sma_w1, sma_w2, sma_w3), support_window)
                    if not chart_df.empty:
                        st.line_chart(chart_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
                        st.line_chart(chart_df[["RSI"]])
                else:
                    st.warning("No chart data available.")

    st.download_button(
        "📥 Download Results",
        preds.to_csv(index=False).encode(),
        "nifty500_signals.csv",
        "text/csv",
    )

st.markdown("⚠ Educational use only — not financial advice.")
