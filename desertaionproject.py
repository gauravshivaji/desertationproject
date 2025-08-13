import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ML
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

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
    # Flatten MultiIndex if needed (for single-ticker frames sometimes returned as MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Ensure 'Close' column exists and is valid
    if "Close" not in df.columns or df["Close"].dropna().empty:
        return pd.DataFrame()

    df = df.copy()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

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

    # Extra ML-friendly features
    for w in (1, 3, 5, 10):
        df[f"Ret_{w}"] = df["Close"].pct_change(w)

    # Distances from SMAs
    for win in sma_windows:
        df[f"Dist_SMA{win}"] = (df["Close"] - df[f"SMA{win}"]) / df[f"SMA{win}"]

    # Slopes
    for col in ["RSI"] + [f"SMA{w}" for w in sma_windows]:
        df[f"{col}_slope"] = df[col].diff()

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

# ----------- RULE-BASED STRATEGY -----------
# ----------- RULE-BASED STRATEGY -----------
def predict_buy_sell_rule(df, rsi_buy=30, rsi_sell=70):
    if df.empty:
        return df
    results = df.copy()

    # Reversal Buy Point
    results["Reversal_Buy"] = (
        (results["RSI"] < rsi_buy) &
        (results["Bullish_Div"]) &
        (np.abs(results["Close"] - results["Support"]) < 0.03 * results["Close"]) &
        (results["Close"] > results["SMA20"])
    )

    # Trend Buy Point
    results["Trend_Buy"] = (
        (results["Close"] > results["SMA20"]) &
        (results["SMA20"] > results["SMA50"]) &
        (results["SMA50"] > results["SMA200"]) &
        (results["RSI"] > 50)
    )

    # Final Buy/Sell
    results["Buy_Point"] = results["Reversal_Buy"] | results["Trend_Buy"]
    results["Sell_Point"] = (
        ((results["RSI"] > rsi_sell) & (results["Bearish_Div"])) |
        (results["Close"] < results["Support"]) |
        ((results["SMA20"] < results["SMA50"]) & (results["SMA50"] < results["SMA200"]))
    )

    return results


# ----------- LABEL GENERATION FROM RULES -----------
def label_from_rule_signals(df, rsi_buy=30, rsi_sell=70):
    signals_df = predict_buy_sell_rule(df, rsi_buy, rsi_sell)
    label = pd.Series(0, index=df.index)  # 0 = Hold
    label[signals_df["Buy_Point"]] = 1
    label[signals_df["Sell_Point"]] = -1
    return label


# ----------- BUILD ML DATASET USING RULE LABELS -----------
def build_ml_dataset_for_tickers(tickers, start, end, sma_periods, support_window, rsi_buy=30, rsi_sell=70):
    X_list, y_list = [], []
    for ticker in tickers:
        df = download_stock_data(ticker, start, end)
        if df.empty:
            continue
        feat = compute_features(df, sma_periods, support_window)
        y = label_from_rule_signals(feat, rsi_buy, rsi_sell)
        X = feat.dropna().copy()
        y = y.loc[X.index]
        if not X.empty:
            X_list.append(X)
            y_list.append(y)
    if X_list:
        return pd.concat(X_list), pd.concat(y_list)
    return pd.DataFrame(), pd.Series(dtype=int)


# ----------- TRAIN RANDOM FOREST -----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_rf_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    st.write("### Model Performance")
    st.text(classification_report(y_test, preds))
    return clf


# ----------- ML SIGNAL PREDICTION -----------
def predict_ml_signals(df, model):
    if df.empty:
        return df
    feat = df.copy().dropna()
    preds = model.predict(feat)
    feat["ML_Buy_Point"] = preds == 1
    feat["ML_Sell_Point"] = preds == -1
    return feat


# ----------- STREAMLIT: ML SIGNALS TAB -----------
    elif strategy == "ML Signals":
    st.subheader("🤖 ML vs Rule-Based Signals")

    sma_w1 = st.sidebar.number_input("SMA Fast", value=20)
    sma_w2 = st.sidebar.number_input("SMA Medium", value=50)
    sma_w3 = st.sidebar.number_input("SMA Slow", value=200)
    support_window = st.sidebar.number_input("Support Window", value=20)
    rsi_buy = st.sidebar.number_input("RSI Buy Level", value=30)
    rsi_sell = st.sidebar.number_input("RSI Sell Level", value=70)

    # Build dataset from all tickers
    X, y = build_ml_dataset_for_tickers(
        tickers, start, end,
        (sma_w1, sma_w2, sma_w3),
        support_window, rsi_buy, rsi_sell
    )

    if X.empty:
        st.error("No data available for ML training.")
    else:
        model = train_rf_classifier(X, y)

        # Predict for selected ticker
        df = download_stock_data(selected_ticker, start, end)
        feat = compute_features(df, (sma_w1, sma_w2, sma_w3), support_window)
        rule_df = predict_buy_sell_rule(feat, rsi_buy, rsi_sell)
        pred_df = predict_ml_signals(feat, model)

        # Merge for comparison
        compare_df = rule_df.join(pred_df[["ML_Buy_Point", "ML_Sell_Point"]])

        # Chart with both
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=compare_df.index, y=compare_df["Close"], mode="lines", name="Close Price"
        ))

        # Rule-based markers
        fig.add_trace(go.Scatter(
            x=compare_df[compare_df["Buy_Point"]].index,
            y=compare_df[compare_df["Buy_Point"]]["Close"],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
            name="Rule Buy"
        ))
        fig.add_trace(go.Scatter(
            x=compare_df[compare_df["Sell_Point"]].index,
            y=compare_df[compare_df["Sell_Point"]]["Close"],
            mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
            name="Rule Sell"
        ))

        # ML-based markers
        fig.add_trace(go.Scatter(
            x=compare_df[compare_df["ML_Buy_Point"]].index,
            y=compare_df[compare_df["ML_Buy_Point"]]["Close"],
            mode="markers", marker=dict(symbol="star", color="blue", size=12),
            name="ML Buy"
        ))
        fig.add_trace(go.Scatter(
            x=compare_df[compare_df["ML_Sell_Point"]].index,
            y=compare_df[compare_df["ML_Sell_Point"]]["Close"],
            mode="markers", marker=dict(symbol="x", color="orange", size=12),
            name="ML Sell"
        ))

        st.plotly_chart(fig, use_container_width=True)
        st.write(compare_df.tail(20))


st.markdown("⚠ Educational use only — not financial advice.")




