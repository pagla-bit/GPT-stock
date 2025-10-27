# streamlit_stock_dashboard_refactor.py
"""
Refactored single-file Streamlit Stock Dashboard
- Parallelized batch fetching
- Cached SPY data
- Plotly candlestick with SMA & Bollinger Bands overlays
- Vectorized Monte Carlo for estimated days to targets
- Session state caching, tunable sim count, weights and thresholds
- Optional VADER sentiment integration (if installed)
- CNN Fear & Greed Index (with fallback attempts)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import requests
import json
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Optional sentiment dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    SENTIMENT_AVAILABLE = True
except Exception:
    SENTIMENT_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Refactored Stock Watch Dashboard")

# -------------------- Utilities & Cached Data --------------------

@st.cache_data
def get_data(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Fetch historical OHLCV data using yfinance with validation.
    Returns tuple (hist_df, info_dict) or (empty_df, {}) on error.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        # Validate full OHLCV structure
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if hist.empty:
            raise ValueError("Empty history returned.")
        missing = set(required_cols) - set(hist.columns)
        if missing:
            raise ValueError(f"Incomplete OHLCV data: missing {missing}")
        if len(hist) < 50:
            raise ValueError("Insufficient historical data (need at least 50 periods)")
        info = tk.info if hasattr(tk, "info") else {}
        return hist, info
    except Exception as e:
        # Return empty frame and an info dict with error for visibility
        return pd.DataFrame(), {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_spy_data(period="1y", interval="1d"):
    """Cache SPY data for correlation calculations."""
    hist, _ = get_data("SPY", period=period, interval=interval)
    return hist

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Try to fetch CNN Fear & Greed for today and last two days. Returns (score, rating, color_label) or (None,'N/A','N/A')"""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0; +https://streamlit.io)"
    }
    base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    for days_back in range(0, 3):
        d = (date.today() - timedelta(days=days_back)).isoformat()
        try:
            resp = requests.get(base_url + d, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score")
            rating = fg.get("rating", "N/A")
            if score is None:
                continue
            if score < 25:
                color = "🟥 Extreme Fear"
            elif score < 45:
                color = "🔴 Fear"
            elif score < 55:
                color = "🟡 Neutral"
            elif score < 75:
                color = "🟢 Greed"
            else:
                color = "🟩 Extreme Greed"
            return score, rating, color
        except Exception:
            continue
    return None, "N/A", "N/A"

# -------------------- Indicator Calculations --------------------

def calc_indicators(df: pd.DataFrame,
                    rsi_period=14,
                    macd_fast=12, macd_slow=26, macd_signal=9,
                    sma_short=20, sma_long=50,
                    bb_period=20, atr_period=14, adx_period=14):
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(sma_short).mean()
    df["SMA_long"] = df["Close"].rolling(sma_long).mean()

    # RSI (EMA-based)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

    # Bollinger Bands
    bb_mid = df["Close"].rolling(bb_period).mean()
    bb_std = df["Close"].rolling(bb_period).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / bb_mid

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ADX (simplified)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / df["ATR"])
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / df["ATR"])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.ewm(span=adx_period).mean()
    return df

def rule_based_signal(df: pd.DataFrame,
                      rsi_oversold=30,
                      rsi_overbought=70,
                      weights=None):
    """
    Weighted rule-based signals. Returns (recommendation_str, signals_list, confidence_percent)
    `weights` is a dict e.g. {'RSI':2.0, 'MACD':1.5, 'SMA':1.0, 'BB':1.0, 'Volume':0.5, 'ADX':1.0}
    """
    if weights is None:
        weights = {'RSI': 2.0, 'MACD': 1.5, 'SMA': 1.0, 'BB': 1.0, 'Volume': 0.5, 'ADX': 1.0}
    if len(df) < 3:
        return "HOLD", [], 0.0

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # RSI
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < rsi_oversold:
            signals.append(("RSI oversold -> BUY", weights['RSI']))
        elif latest['RSI'] > rsi_overbought:
            signals.append(("RSI overbought -> SELL", weights['RSI']))

    # MACD crossover
    if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
        signals.append(("MACD bullish crossover -> BUY", weights['MACD']))
    elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
        signals.append(("MACD bearish crossover -> SELL", weights['MACD']))

    # SMA relative to long SMA
    if latest['Close'] > latest['SMA_long']:
        signals.append(("Price above long SMA -> BULLISH", weights['SMA']))
    else:
        signals.append(("Price below long SMA -> BEARISH", weights['SMA']))

    # Bollinger Bands
    if latest['Close'] < latest['BB_lower']:
        signals.append(("Price below BB lower -> BUY", weights['BB']))
    elif latest['Close'] > latest['BB_upper']:
        signals.append(("Price above BB upper -> SELL", weights['BB']))

    # Volume normalized by ATR
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Volume'].mean()
    atr_norm_factor = (1 + (latest['ATR'] / latest['Close'])) if not np.isnan(latest['ATR']) else 1
    vol_atr_norm = latest['Volume'] / (vol_avg20 * atr_norm_factor) if vol_avg20 and atr_norm_factor else 1
    if vol_atr_norm > 1.5:
        signals.append(("Normalized volume spike -> CONFIRM MOVE", weights['Volume']))

    # ADX trend strength
    if not np.isnan(latest['ADX']) and latest['ADX'] > 25:
        signals.append(("Strong trend (ADX>25) -> AMPLIFY signals", weights['ADX']))
    elif not np.isnan(latest['ADX']):
        signals.append(("Weak trend (ADX<25) -> CAUTION", -weights['ADX'] * 0.5))

    buy_votes = sum(w for s, w in signals if any(k in s for k in ['BUY', 'BULLISH', 'AMPLIFY']))
    sell_votes = sum(w for s, w in signals if any(k in s for k in ['SELL', 'BEARISH']))
    total_weight = sum(abs(w) for s, w in signals) if signals else 1
    confidence = max(-100, min(100, (buy_votes - sell_votes) / total_weight * 100))

    aligned_signals = sum(1 for s, _ in signals if any(k in s for k in ['BUY', 'SELL', 'BULLISH', 'BEARISH']))
    if buy_votes > sell_votes and aligned_signals >= 3:
        recommendation = "STRONG BUY"
    elif buy_votes > sell_votes:
        recommendation = "BUY"
    elif sell_votes > buy_votes and aligned_signals >= 3:
        recommendation = "STRONG SELL"
    elif sell_votes > buy_votes:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return recommendation, signals, float(confidence)

# -------------------- Fast Vectorized Monte Carlo --------------------

def estimate_days_to_target_vectorized(df: pd.DataFrame, current_price: float, target_return: float,
                                       sims: int = 5000, max_days: int = 365):
    """
    Vectorized Monte Carlo: simulate daily returns using historical mean/std (normal approx).
    Returns dict {probability, median_days, 90pct_days}
    """
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None}

    mu, sigma = returns.mean(), returns.std()
    # If sigma=0, short-circuit
    if sigma == 0:
        return {'probability': 0.0, 'median_days': None, '90pct_days': None}

    # Simulate normal daily returns (vectorized)
    # shape: (sims, max_days)
    rand = np.random.normal(loc=mu, scale=sigma, size=(sims, max_days))
    price_paths = current_price * np.cumprod(1 + rand, axis=1)
    threshold = current_price * (1 + target_return)
    hits = price_paths >= threshold
    # find first hit day (1-indexed), otherwise np.nan
    first_hit = np.argmax(hits, axis=1) + 1  # argmax returns 0 if none but we'll mask
    no_hit_mask = ~hits.any(axis=1)
    first_hit = first_hit.astype(float)
    first_hit[no_hit_mask] = np.nan

    valid = ~np.isnan(first_hit)
    prob_reach = valid.mean()
    median_days = float(np.nanmedian(first_hit)) if prob_reach > 0 else None
    pct90 = float(np.nanpercentile(first_hit[valid], 90)) if prob_reach > 0 else None

    return {'probability': prob_reach, 'median_days': median_days, '90pct_days': pct90}

# -------------------- UI Layout & Controls --------------------

st.title("Refactored Stock Watch + Indicator Dashboard")

# Sidebar: Watchlist & Settings
st.sidebar.header("Watchlist & Settings")
group_choice = st.sidebar.radio("Select Market Cap Group",
                                ["Big Cap (>$10B)", "Medium Cap ($1B–$10B)", "Small Cap (<$1B)"])
if group_choice.startswith("Big"):
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA"
elif group_choice.startswith("Medium"):
    default_tickers = "AMD, ADBE, PYPL, SQ, DOCU"
else:
    default_tickers = "SOFI, HOOD, RKT, BB"

tickers_input = st.sidebar.text_area("Tickers (comma separated)", value=default_tickers, height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

lookback = st.sidebar.selectbox("Historical lookback", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Data interval", ["1d", "1wk"], index=0)

# Indicator tunings
st.sidebar.markdown("---")
st.sidebar.header("Indicator Tuning")
rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30)
rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 85, 70)
sma_short = st.sidebar.slider("SMA Short Window", 10, 40, 20)
sma_long = st.sidebar.slider("SMA Long Window", 30, 200, 50)

# Weight tunings for signals
st.sidebar.markdown("**Signal weights**")
w_rsi = st.sidebar.slider("RSI weight", 0.0, 5.0, 2.0, 0.1)
w_macd = st.sidebar.slider("MACD weight", 0.0, 5.0, 1.5, 0.1)
w_sma = st.sidebar.slider("SMA weight", 0.0, 5.0, 1.0, 0.1)
w_bb = st.sidebar.slider("BB weight", 0.0, 5.0, 1.0, 0.1)
w_vol = st.sidebar.slider("Volume weight", 0.0, 2.0, 0.5, 0.1)
w_adx = st.sidebar.slider("ADX weight", 0.0, 3.0, 1.0, 0.1)
weights = {'RSI': w_rsi, 'MACD': w_macd, 'SMA': w_sma, 'BB': w_bb, 'Volume': w_vol, 'ADX': w_adx}

# Monte Carlo settings
st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
sim_count = st.sidebar.select_slider("Simulation count", options=[500, 1000, 2500, 5000, 10000], value=2500)
max_days = st.sidebar.slider("Max days for sim", 90, 730, 365, 30)

# Behavior controls
st.sidebar.markdown("---")
max_workers = st.sidebar.slider("Parallel workers (batch)", 1, 12, 5)
limit_batch = st.sidebar.number_input("Batch limit (max tickers fetched at once)", min_value=1, max_value=50, value=10)
use_sentiment = st.sidebar.checkbox("Enable VADER sentiment (if available)", value=False)
if use_sentiment and not SENTIMENT_AVAILABLE:
    st.sidebar.warning("VADER not available in environment. Install vaderSentiment & nltk for this feature.")

# Buttons
st.sidebar.markdown("---")
refresh_button = st.sidebar.button("Refresh Data / Generate Batch Summary")

# -------------------- Fear & Greed & SPY --------------------

fg_score, fg_rating, fg_color = get_fear_greed_index()
spy_hist = get_spy_data(period=lookback, interval=interval)  # cached

# -------------------- Session State init --------------------
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = {}  # ticker -> (hist, info, timestamp)
if "batch_errors" not in st.session_state:
    st.session_state["batch_errors"] = {}

# -------------------- Batch Fetching & Summary --------------------

st.subheader("Watchlist Summary")

def fetch_and_process(ticker):
    """
    Fetch data for a single ticker and compute summary fields.
    Returns tuple (result_dict or None, error_message or None)
    """
    hist, info = get_data(ticker, period=lookback, interval=interval)
    if hist.empty:
        err_msg = info.get("_error", "Unknown fetch error")
        return None, f"{ticker}: {err_msg}"
    df = calc_indicators(hist, rsi_period=rsi_period, macd_fast=12, macd_slow=26,
                         macd_signal=9, sma_short=sma_short, sma_long=sma_long,
                         bb_period=20, atr_period=14, adx_period=14)
    rec, signals, conf = rule_based_signal(df, rsi_oversold=rsi_oversold,
                                           rsi_overbought=rsi_overbought,
                                           weights=weights)
    # P/E
    pe = info.get("forwardPE") or info.get("trailingPE") or "N/A"
    # SPY corr (align lengths)
    corr = 0.0
    try:
        if not spy_hist.empty:
            min_len = min(len(spy_hist), len(df))
            corr = df['Close'].iloc[-min_len:].corr(spy_hist['Close'].iloc[-min_len:])
            corr = 0.0 if np.isnan(corr) else corr
    except Exception:
        corr = 0.0
    sentiment_val = "N/A"
    if use_sentiment and SENTIMENT_AVAILABLE:
        if "news" in info and isinstance(info['news'], list) and info['news']:
            analyzer = SentimentIntensityAnalyzer()
            scores = []
            for article in info['news'][:5]:
                title = article.get('title', '') or ''
                publisher = article.get('publisher', '') or ''
                scores.append(analyzer.polarity_scores(title + " " + publisher)['compound'])
            if scores:
                sentiment_val = np.mean(scores)
    price_str = f"${df['Close'].iloc[-1]:.2f}"
    out = {
        "Ticker": ticker,
        "Price": price_str,
        "Rec": rec,
        "Conf (%)": f"{conf:.1f}",
        "SPY Corr": f"{corr:.2f}",
        "P/E": pe,
        "Sentiment": f"{sentiment_val:.2f}" if isinstance(sentiment_val, (float, int)) else sentiment_val
    }
    return out, None

batch_results = []
errors = []

if refresh_button:
    # limit number of tickers for performance
    to_fetch = tickers[:int(limit_batch)]
    futures = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        future_to_ticker = {executor.submit(fetch_and_process, t): t for t in to_fetch}
        for fut in as_completed(future_to_ticker):
            res, err = fut.result()
            if err:
                errors.append(err)
            elif res:
                batch_results.append(res)
    # cache errors to session
    st.session_state["batch_errors"] = {t: e for e in errors for t in [e.split(":")[0]]}
    if batch_results:
        df_batch = pd.DataFrame(batch_results).sort_values(by="Rec", ascending=False)
        st.dataframe(df_batch.reset_index(drop=True))
    else:
        st.info("No batch results to show — check errors below.")
    if errors:
        st.error("Some tickers failed to fetch or process. See messages below.")
        for e in errors:
            st.write("- " + e)

# -------------------- Main Single-Stock Inspector --------------------

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Monitored Stocks")
    selected = st.selectbox("Select a ticker to inspect", options=tickers if tickers else ["AAPL"], index=0)
    st.write("Selected:", selected)

with col2:
    st.subheader("Market Sentiment")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if fg_score is not None:
            st.metric("Fear & Greed Score", fg_score)
        else:
            st.metric("Fear & Greed Score", "N/A")
    with c2:
        st.write(f"**{fg_rating}** {fg_color}")
    with c3:
        if fg_score is not None:
            st.progress(fg_score / 100)

# Fetch single selected ticker (with session caching)
cache_key = f"{selected}_{lookback}_{interval}"
if cache_key in st.session_state["data_cache"]:
    hist, info = st.session_state["data_cache"][cache_key]
else:
    hist, info = get_data(selected, period=lookback, interval=interval)
    st.session_state["data_cache"][cache_key] = (hist, info)

if hist.empty:
    st.error(f"No data for {selected}. Reason: {info.get('_error','Unknown')}")
    st.stop()

df = calc_indicators(hist, rsi_period=rsi_period, sma_short=sma_short, sma_long=sma_long)
latest = df.iloc[-1]

# Top-level metrics
price_str = f"${latest['Close']:.2f}"
vol_str = f"{latest['Volume'] / 1_000_000:.2f}M"
market_cap = info.get("marketCap")
if market_cap:
    mc_str = f"${market_cap/1_000_000:,.0f}M"
else:
    mc_str = "N/A"
pe_val = info.get("forwardPE") or info.get("trailingPE") or "N/A"
pe_str = f"{pe_val:.1f}x" if isinstance(pe_val, (int, float)) else pe_val

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", price_str)
m2.metric("Volume", vol_str)
m3.metric("Market Cap", mc_str)
m4.metric("Fwd P/E", pe_str)

# Correlation
corr = 0.0
if not spy_hist.empty:
    try:
        min_len = min(len(spy_hist), len(df))
        corr = df['Close'].iloc[-min_len:].corr(spy_hist['Close'].iloc[-min_len:])
        corr = 0.0 if np.isnan(corr) else corr
    except Exception:
        corr = 0.0
st.metric("SPY Correlation", f"{corr:.2f}")

# Sentiment metric (if enabled)
if use_sentiment and SENTIMENT_AVAILABLE:
    sentiment_score = "N/A"
    if "news" in info and info['news']:
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in info['news'][:5]:
            title = article.get('title','') or ''
            publisher = article.get('publisher','') or ''
            scores.append(analyzer.polarity_scores(title + " " + publisher)['compound'])
        if scores:
            sentiment_score = float(np.mean(scores))
    st.metric("News Sentiment", sentiment_score)

st.markdown("---")
st.subheader("Price Chart & Indicators (interactive)")
# Plotly Candlestick with SMA and BB overlays
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
if 'SMA_short' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_short'], mode='lines', name=f"SMA {sma_short}", line=dict(width=1)))
if 'SMA_long' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_long'], mode='lines', name=f"SMA {sma_long}", line=dict(width=1)))
if 'BB_upper' in df and 'BB_lower' in df:
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], mode='lines', name='BB Upper', line=dict(dash='dot', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], mode='lines', name='BB Lower', line=dict(dash='dot', width=1)))
fig.update_layout(height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Latest indicators table
st.subheader("Latest Indicators")
ind_latest = pd.DataFrame({
    'Value': [
        f"{latest['RSI']:.2f}" if not np.isnan(latest['RSI']) else "N/A",
        f"{latest['MACD']:.6f}" if 'MACD' in latest and not np.isnan(latest['MACD']) else "N/A",
        f"{latest.get('MACD_signal', np.nan):.6f}" if 'MACD_signal' in latest and not np.isnan(latest['MACD_signal']) else "N/A",
        f"{latest['ATR']:.6f}" if not np.isnan(latest['ATR']) else "N/A",
        f"{latest['ADX']:.2f}" if not np.isnan(latest['ADX']) else "N/A",
        f"{latest['BB_width']:.6f}" if not np.isnan(latest.get('BB_width', np.nan)) else "N/A"
    ]
}, index=['RSI','MACD','MACD_signal','ATR','ADX','BB_width'])
st.table(ind_latest)

st.markdown("---")
st.subheader("Recommendation & Monte Carlo holding period estimates")
recommendation, signals, confidence = rule_based_signal(df, rsi_oversold=rsi_oversold,
                                                        rsi_overbought=rsi_overbought,
                                                        weights=weights)
# Display signals
st.write("Algorithmic signals:")
for s, w in signals:
    st.write(f"- {s} (w={w:.2f})")

color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"
st.markdown(f"<h3 style='color:{color}'>{recommendation} (Confidence: {confidence:.1f}%)</h3>", unsafe_allow_html=True)

# Targets and simulations
targets = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
sim_results = []
current_price = float(latest['Close'])
for t in targets:
    res = estimate_days_to_target_vectorized(df, current_price, target_return=t, sims=sim_count, max_days=max_days)
    sim_results.append({
        "Target (%)": int(t*100),
        "Prob. to reach within max days (%)": round(res['probability']*100, 2),
        "Median Days": res['median_days'],
        "90th Pctl Days": res['90pct_days']
    })
st.table(pd.DataFrame(sim_results))

st.markdown("---")
st.subheader("Notes, Limitations & Next Steps")
st.write("""
- This dashboard uses rule-based signals (not financial advice). Backtest rules before deploying.
- Batch fetch is parallelized; reduce batch limit or sim_count if local resources are constrained.
- CNN Fear & Greed is fetched with a user-agent and fallback attempts; network issues may cause unavailability.
- For production: separate modules (data, indicators, UI), persistent caching (Redis), rate-limit handling, and proper API keys for premium data sources are recommended.
""")

st.markdown("### Quick run instructions")
st.code("""
pip install streamlit yfinance pandas numpy plotly requests
# (optional)
pip install vaderSentiment nltk
streamlit run streamlit_stock_dashboard_refactor.py
""")
