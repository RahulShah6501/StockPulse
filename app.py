import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import requests
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

st.set_page_config(page_title="📈 Stock Market Dashboard", layout="wide")

st.markdown("""
# 📈 Advanced Stock Market Dashboard
Use this interactive dashboard to analyze, compare, and forecast stock market trends.
""")

st.sidebar.header("🔧 Configuration")

use_live_data = st.sidebar.checkbox("Use live data from Yahoo Finance", value=True)

if use_live_data:
    symbol_input_main = st.text_input(
        "Enter stock symbols (comma-separated)",
        value="",
        help="Enter any global stock symbol (e.g., AAPL, LLY, NSRGY, TSM)"
    )
    symbol_input_sidebar = st.sidebar.text_input(
        "Optional: Enter stock symbols here (used only if main input is empty)",
        value=""
    )
    input_value = symbol_input_main if symbol_input_main else symbol_input_sidebar

    if input_value:
        symbols = [s.strip().upper() for s in input_value.split(",") if s.strip()]
        df_list = []
        for sym in symbols:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start="2000-01-01", end="2025-12-31")
            if hist.empty:
                st.sidebar.warning(f"No data found for symbol: {sym}")
                continue
            hist = hist.reset_index()
            hist['Index'] = sym
            df_list.append(hist)
        if not df_list:
            st.stop()
        df = pd.concat(df_list, ignore_index=True)
    else:
        st.sidebar.warning("Please enter stock symbols.")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload a stock CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        if 'Index' not in df.columns:
            df['Index'] = 'UploadedData'
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
    else:
        st.sidebar.warning("Upload a CSV file or enable live data.")
        st.stop()

# Clean up date columns
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.tz_localize(None)
df.sort_values(['Index','Date'], inplace=True)

# Multiselect for symbols with no default selected
if df['Index'].nunique() > 1:
    all_symbols = sorted(df['Index'].unique())
    st.subheader("📈 Select Stocks to Compare")
    selected_symbols = st.multiselect(
        "Start typing stock symbols to search and select:",
        options=all_symbols,
        default=[],
        help="Type to get stock suggestions",
        placeholder="Select your stocks"
    )
    if not selected_symbols:
        st.warning("Please select at least one symbol.")
        st.stop()
    df = df[df['Index'].isin(selected_symbols)]
    # Update the sidebar dropdown to only show selected stocks
    selected_detail_symbol = st.sidebar.selectbox("Select one stock for detailed charts:", options=selected_symbols)
else:
    selected_symbols = df['Index'].unique()
    selected_detail_symbol = st.sidebar.selectbox("Select one stock for detailed charts:", options=selected_symbols)

# Date range filter
min_date, max_date = df['Date'].min(), df['Date'].max()
start_date = st.sidebar.date_input("Start Date", min_value=pd.to_datetime("2000-01-01"), max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=pd.to_datetime("2000-01-01"), max_value=max_date, value=max_date)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

if df.empty:
    st.warning("No data available for selected dates.")
    st.stop()

# Add feature engineering
def add_features(group):
    group = group.sort_values('Date').copy()
    group['Returns'] = group['Close'].pct_change()
    group['MA20'] = group['Close'].rolling(window=20).mean()
    group['MA50'] = group['Close'].rolling(window=50).mean()
    group['EMA20'] = group['Close'].ewm(span=20).mean()
    group['STD'] = group['Close'].rolling(window=20).std()
    group['Upper_BB'] = group['MA20'] + (2 * group['STD'])
    group['Lower_BB'] = group['MA20'] - (2 * group['STD'])
    delta = group['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    group['RSI'] = 100 - (100 / (1 + rs))
    group['Volume_Change_%'] = group['Volume'].pct_change() * 100
    group['MA_Crossover'] = np.where(group['MA20'] > group['MA50'], 'Bullish', 'Bearish')
    group['Volatility_30d'] = group['Returns'].rolling(window=30).std() * np.sqrt(252)
    return group

df = df.groupby('Index').apply(add_features).reset_index(drop=True)

# Earnings data placeholder
earnings_data = {}
for sym in selected_symbols:
    ticker = yf.Ticker(sym)
    earnings_df = ticker.earnings
    if earnings_df is not None and not earnings_df.empty:
        earnings_data[sym] = earnings_df.reset_index()
    else:
        earnings_data[sym] = None
        
# KPI Section
st.markdown("""---
### 📌 Key Metrics Overview
""")
kpi_cols = st.columns(len(selected_symbols))
for i, sym in enumerate(selected_symbols):
    sub_df = df[df['Index'] == sym]
    latest = sub_df['Close'].iloc[-1]
    avg = sub_df['Close'].mean()
    change_pct = ((latest - sub_df['Close'].iloc[0]) / sub_df['Close'].iloc[0]) * 100
    kpi_cols[i].metric(f"{sym} Close", f"${latest:.2f}", f"{change_pct:.2f}%")

# Line Chart
st.markdown("""---
### 📈 Price Trends
""")
fig_price = go.Figure()
for sym in selected_symbols:
    data = df[df['Index'] == sym]
    fig_price.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=sym))
fig_price.update_layout(xaxis_title="Date", yaxis_title="Price", template="plotly_white")
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("📊 Volatility (30-day Annualized) Over Time")
fig_vol = go.Figure()
for sym in selected_symbols:
    sym_df = df[df['Index'] == sym]
    fig_vol.add_trace(go.Scatter(x=sym_df['Date'], y=sym_df['Volatility_30d'], mode='lines', name=sym))
fig_vol.update_layout(title="30-Day Annualized Volatility", xaxis_title="Date", yaxis_title="Volatility", template="plotly_white")
st.plotly_chart(fig_vol, use_container_width=True)

# Volume
st.markdown("""---
### 📊 Volume Trends
""")
fig_vol = go.Figure()
for sym in selected_symbols:
    data = df[df['Index'] == sym]
    fig_vol.add_trace(go.Scatter(x=data['Date'], y=data['Volume'], name=f"{sym} Volume"))
fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volume", template="plotly_white")
st.plotly_chart(fig_vol, use_container_width=True)

# MA Crossover Info
st.markdown("""---
### 🔄 MA Crossover Signals
""")
for sym in selected_symbols:
    latest_signal = df[df['Index'] == sym]['MA_Crossover'].iloc[-1]
    st.info(f"**{sym} MA Crossover**: {latest_signal}")

# Correlation
st.markdown("""---
### 🔗 Correlation Heatmap
""")
returns = df.pivot(index='Date', columns='Index', values='Returns')
corr = returns.corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
st.plotly_chart(fig_corr, use_container_width=True)

# RSI
st.markdown("""---
### ⚠️ RSI Alerts
""")
for sym in selected_symbols:
    latest_rsi = df[df['Index'] == sym]['RSI'].iloc[-1]
    if latest_rsi > 70:
        st.warning(f"{sym} RSI = {latest_rsi:.2f} (Overbought)")
    elif latest_rsi < 30:
        st.success(f"{sym} RSI = {latest_rsi:.2f} (Oversold)")
    else:
        st.info(f"{sym} RSI = {latest_rsi:.2f} (Neutral)")

# Sentiment API
st.markdown("""---
### 📰 News Sentiment Analysis
""")
api_key = st.sidebar.text_input("Finnhub API Key", type="password")
if api_key and use_live_data:
    sid = SentimentIntensityAnalyzer()
    sentiment_df_all = []
    for sym in selected_symbols:
        url = f"https://finnhub.io/api/v1/company-news?symbol={sym}&from=2024-01-01&to=2025-12-31&token={api_key}"
        res = requests.get(url)
        if res.status_code == 200:
            news = pd.DataFrame(res.json())
            if not news.empty:
                news['Date'] = pd.to_datetime(news['datetime'], unit='s').dt.date
                news['SentimentScore'] = news['headline'].apply(lambda x: sid.polarity_scores(x)['compound'])
                grouped = news.groupby('Date')['SentimentScore'].mean().reset_index()
                grouped['Index'] = sym
                sentiment_df_all.append(grouped)
    if sentiment_df_all:
        sentiment_all = pd.concat(sentiment_df_all)
        sentiment_all['Date'] = pd.to_datetime(sentiment_all['Date'])
        df = pd.merge(df, sentiment_all, on=['Date', 'Index'], how='left')
        for sym in selected_symbols:
            sub_df = df[df['Index'] == sym].dropna(subset=['SentimentScore'])
            if not sub_df.empty:
                fig_sent = px.line(sub_df, x='Date', y='SentimentScore', title=f"Sentiment Score - {sym}")
                st.plotly_chart(fig_sent, use_container_width=True)

# Prophet forecast per stock
# --- Conservative Forecast (log-returns + reconstruction) ---
st.markdown("""---
### 🔮 Price Forecast (Conservative)
""")

# --- Choose horizon (keeps your existing UI) ---
forecast_horizon = st.sidebar.selectbox(
    "Select Forecast Horizon",
    options=[7, 15, 30, 90],
    index=2,  # default 30
    help="Number of future days to predict"
)

# --- Option to show full table (keeps your existing UI) ---
show_table = st.sidebar.checkbox("Show full forecast table", value=False)

# Helper: forecast returns with Prophet and reconstruct price path
def forecast_by_returns(price_df, periods, clip_pct=0.20):
    """
    price_df: DataFrame with columns ['ds','y'] where y is price
    periods: number of future business days to forecast
    clip_pct: maximum absolute daily return (as fraction, e.g. 0.20 = 20%)
    Returns DataFrame of future dates with predicted prices and bands.
    """
    # Prepare log-returns series
    df = price_df.copy().sort_values('ds')
    df['log_price'] = np.log(df['y'])
    df['y_ret'] = df['log_price'].diff()        # log-return series
    df_ret = df[['ds','y_ret']].dropna().rename(columns={'y_ret':'y'})

    # Require enough history
    if df_ret.shape[0] < 30:
        raise ValueError("Not enough return history (need >=30 rows).")

    # Fit Prophet on returns (stationary)
    m = Prophet(
        growth='flat',
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    m.fit(df_ret)

    # Future dataframe (business days)
    future = m.make_future_dataframe(periods=periods, freq='B')
    forecast = m.predict(future)

    # Keep only future rows (strictly after last observed date)
    last_obs = df_ret['ds'].max()
    fut = forecast[forecast['ds'] > last_obs].copy().reset_index(drop=True)

    # Clip daily log-return forecasts to avoid absurd spikes
    fut['yhat_clipped'] = fut['yhat'].clip(lower=-clip_pct, upper=clip_pct)
    fut['yhat_lower_clipped'] = fut['yhat_lower'].clip(lower=-clip_pct, upper=clip_pct)
    fut['yhat_upper_clipped'] = fut['yhat_upper'].clip(lower=-clip_pct, upper=clip_pct)

    # Reconstruct price path from last actual price
    last_price = float(price_df['y'].iloc[-1])
    fut['cum_loghat'] = fut['yhat_clipped'].cumsum()
    fut['cum_loglower'] = fut['yhat_lower_clipped'].cumsum()
    fut['cum_logupper'] = fut['yhat_upper_clipped'].cumsum()

    fut['pred_price'] = last_price * np.exp(fut['cum_loghat'])
    fut['pred_lower'] = last_price * np.exp(fut['cum_loglower'])
    fut['pred_upper'] = last_price * np.exp(fut['cum_logupper'])

    # safety floor
    fut['pred_price'] = fut['pred_price'].clip(lower=0.01)
    fut['pred_lower'] = fut['pred_lower'].clip(lower=0.01)
    fut['pred_upper'] = fut['pred_upper'].clip(lower=0.01)

    return fut[['ds','pred_price','pred_lower','pred_upper','yhat_clipped','yhat_lower_clipped','yhat_upper_clipped']]

# Run forecasts for each selected symbol
for sym in selected_symbols:
    base = df[df['Index'] == sym].copy()
    price_col = 'Adj Close' if 'Adj Close' in base.columns else 'Close'

    # Respect your earlier requirement: start from 2025-01-01 by default (but still respects top-level date filter)
    sub_prices = base[base['Date'] >= pd.to_datetime("2025-01-01")][['Date', price_col]] \
        .dropna().rename(columns={'Date':'ds', price_col:'y'}).reset_index(drop=True)

    if sub_prices.empty or len(sub_prices) < 60:
        st.info(f"Not enough data to forecast {sym}. Need ~60 days after 2025-01-01 or adjust your start date.")
        continue

    try:
        fut = forecast_by_returns(sub_prices, periods=forecast_horizon, clip_pct=0.20)
    except Exception as e:
        st.warning(f"Forecast skipped for {sym}: {e}")
        continue

    # Present results (vertical metrics)
    st.subheader(f"📌 {sym}")
    last_actual = sub_prices['y'].iloc[-1]
    next_day_pred = float(fut['pred_price'].iloc[0])
    end_horizon_pred = float(fut['pred_price'].iloc[-1])

    # Vertical display (compact, non-technical)
    st.markdown(f"**Latest Price:**  \n${last_actual:,.2f}")
    st.markdown(f"**Predicted Next-Day Price:**  \n${next_day_pred:,.2f}")
    st.markdown(f"**Predicted Price after {forecast_horizon} days:**  \n${end_horizon_pred:,.2f}")

    # Chart: actual prices + predicted price path with band
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub_prices['ds'], y=sub_prices['y'],
                             mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=fut['ds'], y=fut['pred_price'],
                             mode='lines', name='Predicted Price'))
    fig.add_trace(go.Scatter(
        x=list(fut['ds']) + list(fut['ds'][::-1]),
        y=list(fut['pred_upper']) + list(fut['pred_lower'][::-1]),
        fill='toself', line=dict(width=0), opacity=0.2, name='Prediction Range (95%)'
    ))
    fig.update_layout(title=f"{sym} {forecast_horizon}-Day Forecast (returns-based)",
                      xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Optional compact table (only the future horizon)
    if show_table:
        table = fut[['ds','pred_price']].rename(columns={'ds':'Date','pred_price':'Predicted Price ($)'}).round(2)
        st.dataframe(table, use_container_width=True)

# Detailed charts for selected stock
st.markdown("""---
### 🔍 Detailed Technical Charts
""")
detail_df = df[df['Index'] == selected_detail_symbol]

# Candlestick
st.info("Candlestick chart with OHLC data")
fig_candle = go.Figure(data=[go.Candlestick(x=detail_df['Date'], open=detail_df['Open'], high=detail_df['High'],
                                            low=detail_df['Low'], close=detail_df['Close'])])
fig_candle.update_layout(title=f"{selected_detail_symbol} Candlestick", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_candle, use_container_width=True)

st.subheader(f"💰 Earnings (EPS) Over Years - {selected_detail_symbol}")

earn_df = earnings_data.get(selected_detail_symbol)

if earn_df is not None and not earn_df.empty:
    fig_eps = go.Figure()
    fig_eps.add_trace(go.Bar(x=earn_df['Year'], y=earn_df['Earnings'], name='EPS'))
    fig_eps.update_layout(title=f"Earnings Per Share (EPS) for {selected_detail_symbol}",
                          xaxis_title="Year", yaxis_title="EPS")
    st.plotly_chart(fig_eps, use_container_width=True)

    st.markdown("""
    **Earnings Summary:**  
    - Positive EPS indicates profitability.  
    - Observe EPS growth trends for company performance.
    """)
else:
    st.info(f"No earnings data available for {selected_detail_symbol}.")

# MA and Bollinger Bands
st.info("Moving Averages and Bollinger Bands")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['Close'], name='Close'))
fig_ma.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['MA20'], name='MA20'))
fig_ma.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['MA50'], name='MA50'))
fig_ma.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['Upper_BB'], name='Upper BB', line=dict(dash='dot')))
fig_ma.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['Lower_BB'], name='Lower BB', line=dict(dash='dot')))
fig_ma.update_layout(xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_ma, use_container_width=True)

# RSI Chart
st.info("RSI Chart")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=detail_df['Date'], y=detail_df['RSI'], name='RSI', line=dict(color='orange')))
fig_rsi.update_layout(yaxis=dict(title='RSI', range=[0, 100]), xaxis_title='Date')
st.plotly_chart(fig_rsi, use_container_width=True)

# Download
st.markdown("""---
### 📥 Download Data
""")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
