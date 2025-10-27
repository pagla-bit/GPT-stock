# streamlit_stock_dashboard.py
# A self-contained Streamlit app that:
# - Lets you add a watchlist of stocks
# - Click/select a stock to view indicators (Market Cap, Volume, RSI, MACD, SMA)
# - Produces a buy/sell recommendation based on a simple rule-based algorithm
# - Estimates tentative holding-period (days) to reach target profits using Monte Carlo

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(layout='wide', page_title='Stock Watch & Signal Dashboard')

# ------------------------- Helper functions -------------------------

def get_data(ticker, period='1y', interval='1d'):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False)
        info = tk.info if hasattr(tk, 'info') else {}
        return hist, info
    except Exception as e:
        st.error(f'Error fetching data for {ticker}: {e}')
        return pd.DataFrame(), {}


def calc_indicators(df):
    df = df.copy()
    # SMA
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    # RSI (14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


def rule_based_signal(df):
    # Uses latest values to create a simple recommendation
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []

    # RSI rules
    if latest['RSI'] < 30:
        signals.append('RSI oversold -> BUY')
    elif latest['RSI'] > 70:
        signals.append('RSI overbought -> SELL')

    # MACD crossover
    if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
        signals.append('MACD bullish crossover -> BUY')
    elif (prev['MACD'] > prev['MACD_signal']) and (latest['MACD'] < latest['MACD_signal']):
        signals.append('MACD bearish crossover -> SELL')

    # Price vs SMA
    if latest['Close'] > latest['SMA50']:
        signals.append('Price above 50-day SMA -> BULLISH')
    else:
        signals.append('Price below 50-day SMA -> BEARISH')

    # Volume spike
    vol_avg20 = df['Volume'].rolling(20).mean().iloc[-1]
    if latest['Volume'] > 1.5 * vol_avg20:
        signals.append('Volume spike -> CONFIRMS recent move')

    # Final simple recommendation logic
    buy_votes = sum(1 for s in signals if 'BUY' in s or 'BULLISH' in s)
    sell_votes = sum(1 for s in signals if 'SELL' in s or 'BEARISH' in s)
    if buy_votes > sell_votes:
        recommendation = 'BUY'
    elif sell_votes > buy_votes:
        recommendation = 'SELL'
    else:
        recommendation = 'HOLD'

    return recommendation, signals


def estimate_days_to_target(df, current_price, target_return, sims=5000, max_days=365):
    # Calibrate GBM from recent daily returns
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    # Simulate many paths of daily returns
    days_to_target = []
    for _ in range(sims):
        price = current_price
        for day in range(1, max_days+1):
            # draw daily return from normal with mu and sigma
            r = np.random.normal(loc=mu, scale=sigma)
            price *= (1 + r)
            if (price / current_price - 1) >= target_return:
                days_to_target.append(day)
                break
        else:
            # didn't reach target within max_days
            days_to_target.append(np.nan)
    # summarize
    days_arr = np.array(days_to_target, dtype=float)
    prob_reach = np.sum(~np.isnan(days_arr)) / sims
    median_days = np.nanmedian(days_arr)
    perc90 = np.nanpercentile(days_arr[~np.isnan(days_arr)], 90) if np.sum(~np.isnan(days_arr))>0 else np.nan
    return {'probability': prob_reach, 'median_days': float(median_days) if not np.isnan(median_days) else None, '90pct_days': float(perc90) if not np.isnan(perc90) else None}

# ------------------------- Streamlit UI -------------------------

st.title('Stock Watch + Indicator Dashboard')

with st.sidebar:
    st.header('Watchlist')
    tickers_input = st.text_area('Enter tickers (comma separated)', value='AAPL, MSFT, GOOGL')
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    st.markdown('---')
    st.header('Settings')
    lookback = st.selectbox('Historical lookback', options=['6mo','1y','2y','5y'], index=1)
    interval = st.selectbox('Data interval', options=['1d','1wk'], index=0)
    run_button = st.button('Refresh data')

# quick guard
if not tickers:
    st.info('Add some tickers in the left sidebar to start.')
    st.stop()

# main layout
col1, col2 = st.columns([1,2])

with col1:
    st.subheader('Monitored stocks')
    selected = st.selectbox('Select a stock', options=tickers)
    st.write('Selected:', selected)

with col2:
    st.subheader('Quick summary (latest)')
    hist, info = get_data(selected, period=lookback, interval=interval)
    if hist.empty:
        st.error('No data. Try another ticker or interval.')
        st.stop()
    df = calc_indicators(hist)
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric('Price', f"{latest['Close']:.2f}")
    c2.metric('Volume', int(latest['Volume']))
    market_cap = info.get('marketCap', None)
    c3.metric('Market Cap', f"{market_cap:,}" if market_cap else 'N/A')

st.markdown('---')

# Charts and indicators
st.subheader('Price chart & Indicators')
price_col, ind_col = st.columns([3,1])
with price_col:
    st.line_chart(df['Close'])
    st.write('SMA20 and SMA50:')
    st.line_chart(df[['SMA20','SMA50']].dropna())
with ind_col:
    st.write('Latest indicators')
    st.write(pd.DataFrame({
        'Value': [f"{latest['RSI']:.2f}", f"{latest['MACD']:.4f}", f"{latest['MACD_signal']:.4f}"]
    }, index=['RSI','MACD','MACD_signal']))

st.markdown('---')

# Recommendation + holding period estimates
st.subheader('Recommendation & Estimated holding periods')
recommendation, signals = rule_based_signal(df)
st.write('Algorithmic signals:')
for s in signals:
    st.write('- ' + s)
st.success(f'Recommendation: {recommendation}')

targets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.00]  # 5% to 100%
results = []
current_price = latest['Close']
for t in targets:
    res = estimate_days_to_target(df, current_price, target_return=t, sims=2000, max_days=365)
    results.append({'target_pct': int(t*100), 'prob_reach_1y': round(res['probability']*100,2), 'median_days': res['median_days'], '90pct_days': res['90pct_days']})

st.table(pd.DataFrame(results))

st.markdown('---')

st.write('Notes and limitations:')
st.write('- This app uses a simple rule-based signal engine and Monte Carlo with historical daily returns to estimate time-to-targets. These are **probabilistic estimates**, not guarantees.')
st.write('- Tune algorithm thresholds (RSI levels, MACD logic) and simulation settings for your risk profile.')

st.write('To run:')
st.code("""
# Install dependencies
pip install streamlit yfinance
# optional: pip install pandas_ta
# Run
streamlit run streamlit_stock_dashboard.py
""")

st.write('If you want, I can:')
st.write('- Convert this into a full backtest to compute historical accuracy of signals')
st.write('- Add more indicators (ATR, Bollinger Bands, OBV, Fundamental ratios)')
st.write('- Prepare a deployment guide (Streamlit Community Cloud, Heroku alternatives)')
