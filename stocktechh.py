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

# @st.cache_data
# def fetch_stock_data(stock_ticker, period="5y"):
#     """Fetch stock data using Yahoo Finance with error handling."""
#     try:
#         df = yf.download(stock_ticker, period=period)
#         if df.empty or df["Close"].isnull().all():
#             raise Exception(f"{selected_stock} stock data is unavailable for this period.")
#         return df
#     except Exception as e:
#         st.error(f"Error fetching {selected_stock} data: {e}")
#         return None

@st.cache_data
def fetch_stock_data(stock_ticker, period="5y"):
    """Fetch stock data using Yahoo Finance with error handling."""
    try:
        df = yf.download(stock_ticker, period=period)
        
        if df is None or df.empty:
            st.error(f"No data found for {selected_stock}. Try a different time period or stock.")
            return None

        if "Close" not in df.columns:
            st.error(f"No 'Close' price data available for {selected_stock}.")
            return None

        if df["Close"].isna().all():
            st.error(f"{selected_stock} stock data is unavailable for this period.")
            return None
        
        # Debug: Print DataFrame info
        st.write("Fetched Data Sample:")
        st.write(df.head())

        # Fill missing values
        df = df.fillna(method="ffill").dropna()

        return df
    except Exception as e:
        st.error(f"Error fetching {selected_stock} data: {e}")
        return None



def calculate_technical_indicators(df):
    """Calculate various technical indicators."""
    # Simple Moving Averages
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # MACD
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'])
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    return df

def plot_moving_averages(df):
    """Plot price with Simple Moving Averages."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200'))
    
    fig.update_layout(
        title='Moving Averages Analysis',
        yaxis_title='Price',
        xaxis_title='Date'
    )
    return fig

def plot_macd(df):
    """Plot MACD indicator."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'], name='MACD'))
    
    fig.update_layout(
        title='MACD Analysis',
        yaxis_title='MACD',
        xaxis_title='Date'
    )
    return fig

def plot_rsi(df):
    """Plot RSI indicator."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
    
    # Add RSI levels
    fig.add_hline(y=70, line_color="red", line_dash="dash")
    fig.add_hline(y=30, line_color="green", line_dash="dash")
    
    fig.update_layout(
        title='RSI Analysis',
        yaxis_title='RSI',
        xaxis_title='Date'
    )
    return fig

def plot_bollinger_bands(df):
    """Plot Bollinger Bands."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band'))
    
    fig.update_layout(
        title='Bollinger Bands Analysis',
        yaxis_title='Price',
        xaxis_title='Date'
    )
    return fig

# Load and process data
data_load_state = st.text("Loading data...")
df = fetch_stock_data(stocks[selected_stock])

if df is None:
    st.error(f"Skipping analysis as no valid stock data is available for {selected_stock}.")
    st.stop()

# Calculate technical indicators
df = calculate_technical_indicators(df)

# Prepare Prophet data
df_prophet = df.reset_index()[["Date", "Close"]]
df_prophet.columns = ["ds", "y"]

data_load_state.text("Loading data... done!")

# Technical Analysis Section
st.header("Technical Analysis")

# Moving Averages
st.subheader("Moving Averages Analysis")
st.plotly_chart(plot_moving_averages(df))
st.write("""
* SMA20 (Blue): Short-term trend
* SMA50 (Orange): Medium-term trend
* SMA200 (Green): Long-term trend
* When shorter SMA crosses above longer SMA: Potential bullish signal
* When shorter SMA crosses below longer SMA: Potential bearish signal
""")

# MACD
st.subheader("MACD Analysis")
st.plotly_chart(plot_macd(df))
st.write("""
* MACD above zero: Bullish momentum
* MACD below zero: Bearish momentum
* MACD crossing zero: Potential trend change
""")

# RSI
st.subheader("RSI Analysis")
st.plotly_chart(plot_rsi(df))
st.write("""
* RSI above 70: Potentially overbought
* RSI below 30: Potentially oversold
* RSI trending: Can indicate momentum
""")

# Bollinger Bands
st.subheader("Bollinger Bands Analysis")
st.plotly_chart(plot_bollinger_bands(df))
st.write("""
* Price near upper band: Potentially overbought
* Price near lower band: Potentially oversold
* Price outside bands: Strong trend
* Bands squeezing: Potential breakout coming
""")

# Prophet Forecast Section
st.header("Price Forecast")

# Fit Prophet model
m = Prophet()
m.fit(df_prophet)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast results
st.subheader("Forecast data")
st.write(forecast.tail())

st.write(f"Forecast plot for {n_years} years")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# Add summary statistics
st.header("Summary Statistics")
summary_data = {
    "Current Price": df['Close'].iloc[-1],
    "20-Day SMA": df['SMA20'].iloc[-1],
    "50-Day SMA": df['SMA50'].iloc[-1],
    "200-Day SMA": df['SMA200'].iloc[-1],
    "RSI": df['RSI'].iloc[-1],
    "Upper Bollinger Band": df['BB_upper'].iloc[-1],
    "Lower Bollinger Band": df['BB_lower'].iloc[-1]
}

st.write(pd.Series(summary_data).round(2))
