import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Stock Forecast App')

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

@st.cache_data  # Fixes caching issues
def fetch_stock_data(stock_ticker, period="5y"):
    """Fetch stock data using Yahoo Finance with error handling."""
    try:
        df = yf.download(stock_ticker, period=period)
        if df.empty or df["Close"].isnull().all():
            raise Exception(f"{selected_stock} stock data is unavailable for this period.")
        df = df.reset_index()[["Date", "Close"]]
        df.columns = ["ds", "y"]  # Prophet requires columns 'ds' and 'y'
        return df
    except Exception as e:
        st.error(f"Error fetching {selected_stock} data: {e}")
        return None

# Load data
data_load_state = st.text("Loading data...")
df_train = fetch_stock_data(stocks[selected_stock])

if df_train is None:
    st.error(f"Skipping forecast as no valid stock data is available for {selected_stock}.")
    st.stop()

data_load_state.text("Loading data... done!")

# Show raw data
st.subheader("Raw data")
st.write(df_train.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], name="Stock Close Price"))
    fig.layout.update(title_text="Time Series Data with Rangeslider", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting using Prophet
m = Prophet()
m.fit(df_train)

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
