import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import ta

st.title('Stock Analysis & Forecast App')

# Define available stock options
stocks = {
    "Google": "GOOGL",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Tata Motors": "TATAMOTORS.NS"
}

# Streamlit UI elements
selected_stock = st.selectbox("Select dataset for prediction", list(stocks.keys()))
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def fetch_stock_data(stock_ticker, period="5y"):
    """Fetch stock data using Yahoo Finance with error handling."""
    try:
        df = yf.download(stock_ticker, period=period)
        if df is None or df.empty:
            st.error(f"No data found for {selected_stock}.")
            return None
        df = df.fillna(method="ffill").dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching {selected_stock} data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators."""
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['Buy'] = (df['SMA20'] > df['SMA50']) & (df['RSI'] < 30)
    df['Sell'] = (df['SMA20'] < df['SMA50']) & (df['RSI'] > 70)
    return df

def plot_moving_averages(df):
    """Plot price with Simple Moving Averages and buy/sell signals."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200'))
    fig.add_trace(go.Scatter(x=df[df['Buy']].index, y=df[df['Buy']]['Close'], mode='markers', marker=dict(color='green', size=8), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=df[df['Sell']].index, y=df[df['Sell']]['Close'], mode='markers', marker=dict(color='red', size=8), name='Sell Signal'))
    return fig

def plot_macd(df):
    """Plot MACD indicator."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'], name='MACD'))
    return fig

def plot_rsi(df):
    """Plot RSI indicator."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    fig.add_hline(y=70, line_color="red", line_dash="dash")
    fig.add_hline(y=30, line_color="green", line_dash="dash")
    fig.add_trace(go.Scatter(x=df[df['Buy']].index, y=df[df['Buy']]['RSI'], mode='markers', marker=dict(color='green', size=8), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=df[df['Sell']].index, y=df[df['Sell']]['RSI'], mode='markers', marker=dict(color='red', size=8), name='Sell Signal'))
    return fig

def plot_bollinger_bands(df):
    """Plot Bollinger Bands."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band'))
    return fig

df = fetch_stock_data(stocks[selected_stock])
if df is None:
    st.stop()
df = calculate_technical_indicators(df)
df_prophet = df.reset_index()[["Date", "Close"]]
df_prophet.columns = ["ds", "y"]

st.header("Technical Analysis")
st.subheader("Moving Averages Analysis")
st.plotly_chart(plot_moving_averages(df))
st.subheader("MACD Analysis")
st.plotly_chart(plot_macd(df))
st.subheader("RSI Analysis")
st.plotly_chart(plot_rsi(df))
st.subheader("Bollinger Bands Analysis")
st.plotly_chart(plot_bollinger_bands(df))

st.header("Price Forecast")
m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.plotly_chart(plot_plotly(m, forecast))
