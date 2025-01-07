import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper import fetch_periods_intervals, fetch_stock_history, generate_stock_prediction
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

##### Sidebar #####
st.sidebar.title("User Input Features")

# Stock ticker input
stock_ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="AAPL")

# Period and interval selection
periods = fetch_periods_intervals()
selected_period = st.sidebar.selectbox("Select Period", list(periods.keys()), index=4)
selected_interval = st.sidebar.selectbox("Select Interval", periods[selected_period])

##### Main Page #####
st.title("📈 Stock Price Prediction")
st.subheader("Enhance investment decisions with data-driven forecasting")

# Fetch historical stock data
stock_data = fetch_stock_history(stock_ticker, selected_period, selected_interval)

if not stock_data.empty:
    st.markdown("### Historical Data")
   
    # Plot historical data
    fig = go.Figure(
        data=[go.Candlestick(
                x=stock_data.index,
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Generate predictions
    train_df, test_df, val_df, next_day_price, y_train_date, y_test_date, y_val_date = generate_stock_prediction(stock_ticker)

    # Add tomorrow's date for next day prediction
    tomorrow = datetime.now() + timedelta(days=1)

    if next_day_price is not None:
        st.markdown(f"### Predicted Price for {tomorrow.strftime('%Y-%m-%d')}: ${next_day_price:.2f}")
    else:
        st.warning("Unable to predict the next day's stock price.")

    if train_df is not None and test_df is not None and val_df is not None:
        # Function to plot data
        def plot_predictions(df, date_series, title):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=date_series,
                y=df['Actual'],
                mode='lines',
                name='Actual'
            ))
            fig.add_trace(go.Scatter(
                x=date_series,
                y=df['Predicted'],
                mode='lines',
                name='Predicted'
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Stock Price",
                template="plotly_white"
            )
            return fig

        # Plot training data
        st.markdown("## **Stock Prediction - Training Data**")
        st.plotly_chart(plot_predictions(train_df, y_train_date, "Training Data vs. Predictions"), use_container_width=True)

        # Plot validation data
        st.markdown("## **Stock Prediction - Validation Data**")
        st.plotly_chart(plot_predictions(val_df, y_val_date, "Validation Data vs. Predictions"), use_container_width=True)

        # Plot test data
        st.markdown("## **Stock Prediction - Test Data**")
        st.plotly_chart(plot_predictions(test_df, y_test_date, "Test Data vs. Predictions"), use_container_width=True)
    else:
        st.warning("Unable to generate predictions for the selected stock.")
else:
    st.error("No historical data available for the provided ticker and period/interval.")
