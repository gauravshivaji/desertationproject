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

    # Trend Buy Point: Triple SMA alignment
    results["Trend_Buy"] = (
        (results["Close"] > results["SMA20"]) &
        (results["SMA20"] > results["SMA50"]) &
        (results["SMA50"] > results["SMA200"]) &
        (results["RSI"] > 50)
    )

    # Final Buy Point
    results["Sell_Point"] = results["Reversal_Buy"] | results["Trend_Buy"]

    # Sell Point logic
    results["Buy_Point"] = (
        ((results["RSI"] > rsi_sell) & (results["Bearish_Div"])) |
        (results["Close"] < results["Support"]) |
        ((results["SMA20"] < results["SMA50"]) & (results["SMA50"] < results["SMA200"]))
    )

    return results

# ----------- ML PIPELINE -----------
@st.cache_data(show_spinner=False)
def load_history_for_ticker(ticker, period="5y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def label_from_future_returns(df, horizon=5, buy_thr=0.03, sell_thr=-0.03):
    """Create 3-class label using future return over `horizon` days."""
    fut_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = pd.Series(0, index=df.index)  # 0 = Hold
    label[fut_ret >= buy_thr] = 1         # 1 = Buy
    label[fut_ret <= sell_thr] = -1       # -1 = Sell
    return label

def build_ml_dataset_for_tickers(tickers, sma_windows, support_window, horizon, buy_thr, sell_thr, min_rows=250):
    """Pool data across selected tickers for training."""
    X_list, y_list, meta_list = [], [], []
    feature_cols = None

    for t in tickers:
        hist = load_history_for_ticker(t, period="5y", interval="1d")
        if hist is None or hist.empty or len(hist) < min_rows:
            continue

        feat = compute_features(hist, sma_windows, support_window)
        if feat.empty:
            continue

        # Labeling
        y = label_from_future_returns(feat, horizon=horizon, buy_thr=buy_thr, sell_thr=sell_thr)

        # Drop rows with NaNs and align
        data = feat.join(y.rename("Label")).dropna()
        if data.empty:
            continue

        # Feature selection (exclude leakage)
        drop_cols = set(["Label"])
        drop_cols |= set(["Support","Bullish_Div","Bearish_Div"])  # optional; can keep if desired
        # Keep numeric columns only
        use = data.select_dtypes(include=[np.number]).drop(columns=list(drop_cols.intersection(data.columns)), errors="ignore")

        # Define feature columns once
        if feature_cols is None:
            feature_cols = list(use.columns)

        X_list.append(use[feature_cols])
        y_list.append(data["Label"])
        meta_list.append(pd.Series([t] * len(use), index=use.index, name="Ticker"))

    if not X_list:
        return pd.DataFrame(), pd.Series(dtype=int), [], []

    X = pd.concat(X_list, axis=0)
    y = pd.concat(y_list, axis=0)
    tickers_series = pd.concat(meta_list, axis=0)

    return X, y, feature_cols, tickers_series

def train_rf_classifier(X, y, random_state=42):
    """Train RandomForest with class balancing; simple holdout for speed."""
    # Guard
    if X.empty or y.empty:
        return None, None, None

    # Stratify may fail if any class has <2 samples; fall back gracefully
    stratify_opt = y if len(np.unique(y)) > 1 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=stratify_opt, random_state=42
        )
    except Exception:
        X_train, X_test, y_train, y_test = X.iloc[:-200], X.iloc[-200:], y.iloc[:-200], y.iloc[-200:]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Quick metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

    return clf, acc, report

def latest_feature_row_for_ticker(ticker, sma_windows, support_window, feature_cols):
    """Compute latest row of ML features for a ticker."""
    hist = load_history_for_ticker(ticker, period="3y", interval="1d")
    if hist is None or hist.empty:
        return None
    feat = compute_features(hist, sma_windows, support_window).dropna()
    if feat.empty:
        return None
    use = feat.select_dtypes(include=[np.number])
    row = use.iloc[-1:]

    # Ensure same feature columns order
    missing = [c for c in feature_cols if c not in row.columns]
    for m in missing:
        row[m] = 0.0
    row = row[feature_cols]
    return row

# ----------- UI -----------
st.set_page_config(page_title="Nifty500 Buy/Sell Predictor", layout="wide")
st.title("📊 Nifty500 Buy/Sell Predictor (Rule-based + 🤖 ML)")

with st.sidebar:
    st.header("Settings")
    selected_tickers = st.multiselect(
        "Select stocks", NIFTY500_TICKERS, default=NIFTY500_TICKERS[:5]
    )
    sma_w1 = st.number_input("SMA Window 1", 5, 250, 20)
    sma_w2 = st.number_input("SMA Window 2", 5, 250, 50)
    sma_w3 = st.number_input("SMA Window 3", 5, 250, 200)
    support_window = st.number_input("Support Period (days)", 5, 90, 30)

    st.markdown("---")
    st.subheader("Rule-based thresholds")
    rsi_buy = st.slider("RSI Buy Threshold", 10, 50, 30)
    rsi_sell = st.slider("RSI Sell Threshold", 50, 90, 70)

    st.markdown("---")
    st.subheader("ML labeling (future return)")
    ml_horizon = st.number_input("Horizon (days ahead)", 2, 20, 5)
    ml_buy_thr = st.number_input("Buy threshold (e.g., 0.03 = +3%)", 0.005, 0.20, 0.03, step=0.005, format="%.3f")
    ml_sell_thr = st.number_input("Sell threshold (e.g., -0.03 = -3%)", -0.20, -0.005, -0.03, step=0.005, format="%.3f")

    run_analysis = st.button("Run Analysis")

if run_analysis:
    sma_tuple = (sma_w1, sma_w2, sma_w3)

    with st.spinner("Fetching data & computing rule-based features..."):
        feats = get_features_for_all(selected_tickers, sma_tuple, support_window)
        if feats.empty:
            st.error("No valid data for selected tickers.")
        else:
            preds_rule = predict_buy_sell_rule(feats, rsi_buy, rsi_sell)

    tab1, tab2, tab3, tab4 = st.tabs(["✅ Rule Buy", "❌ Rule Sell", "📈 Charts", "🤖 ML Signals"])

    # -------- Rule-based tabs --------
    with tab1:
        if feats.empty:
            st.info("No rule-based buy signals.")
        else:
            st.dataframe(preds_rule[preds_rule["Buy_Point"]])

    with tab2:
        if feats.empty:
            st.info("No rule-based sell signals.")
        else:
            st.dataframe(preds_rule[preds_rule["Sell_Point"]])

    with tab3:
        ticker_for_chart = st.selectbox("Chart Ticker", selected_tickers)
        chart_df = yf.download(
            ticker_for_chart, period="6mo", interval="1d", progress=False
        )
        if not chart_df.empty:
            chart_df = compute_features(chart_df, sma_tuple, support_window)
            if not chart_df.empty:
                st.line_chart(chart_df[["Close", f"SMA{sma_w1}", f"SMA{sma_w2}", f"SMA{sma_w3}"]])
                st.line_chart(chart_df[["RSI"]])
        else:
            st.warning("No chart data available.")

    # -------- ML tab --------
    with tab4:
        if not SKLEARN_OK:
            st.error("scikit-learn not available. Install with: pip install scikit-learn")
        else:
            with st.spinner("Building ML dataset & training model..."):
                X, y, feature_cols, tickers_series = build_ml_dataset_for_tickers(
                    selected_tickers, sma_tuple, support_window,
                    horizon=ml_horizon, buy_thr=ml_buy_thr, sell_thr=ml_sell_thr
                )

                if X.empty or y.empty:
                    st.warning("Not enough historical data to train the ML model for the chosen settings.")
                else:
                    clf, acc, report = train_rf_classifier(X, y)
                    st.caption(f"Validation accuracy (holdout): **{acc:.3f}**")
                    with st.expander("Classification report"):
                        st.text(report)

                    # Predict latest for each selected ticker
                    rows = []
                    for t in selected_tickers:
                        row = latest_feature_row_for_ticker(t, sma_tuple, support_window, feature_cols)
                        if row is None:
                            continue
                        proba = clf.predict_proba(row)[0] if hasattr(clf, "predict_proba") else None
                        pred = clf.predict(row)[0]
                        rows.append({
                            "Ticker": t,
                            "ML_Pred": {1: "BUY", 0: "HOLD", -1: "SELL"}.get(int(pred), "HOLD"),
                            "Prob_Buy": float(proba[list(clf.classes_).index(1)]) if proba is not None and 1 in clf.classes_ else np.nan,
                            "Prob_Hold": float(proba[list(clf.classes_).index(0)]) if proba is not None and 0 in clf.classes_ else np.nan,
                            "Prob_Sell": float(proba[list(clf.classes_).index(-1)]) if proba is not None and -1 in clf.classes_ else np.nan,
                        })
                    if rows:
                        ml_df = pd.DataFrame(rows).sort_values(["ML_Pred","Prob_Buy"], ascending=[True, False])
                        st.dataframe(ml_df, use_container_width=True)
                    else:
                        st.info("Could not compute ML features for the selected tickers.")

    # -------- Download --------
    if 'preds_rule' in locals() and not preds_rule.empty:
        st.download_button(
            "📥 Download Rule-based Results",
            preds_rule.to_csv(index=False).encode(),
            "nifty500_rule_signals.csv",
            "text/csv",
        )

st.markdown("⚠ Educational use only — not financial advice.")


