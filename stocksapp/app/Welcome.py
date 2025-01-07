import streamlit as st

# Configure the page
st.set_page_config(
    page_title="📈 Stock Price Prediction App",
    page_icon="🌟",
    layout="wide"
)

# Main Page
st.title("📈✨ **Stock Price Prediction App** ✨")
st.subheader("🎉 Welcome to the **Stock Price Prediction Application**! 🎉")

# About the App
st.markdown(
    """
    <div style="text-align: center;">
        <h3>🌟 About this App 🌟</h3>
        <p>
        This application allows users to:
        </p>
        <ul style="list-style-type: none;">
            <li>🔍 View historical stock data.</li>
            <li>📊 Visualize stock price trends with interactive candlestick charts.</li>
            <li>🤖 Predict future stock prices using machine learning models.</li>
        </ul>
        <p>
        The predictions are based on time series data and cutting-edge deep learning techniques. 
        The app supports various periods and intervals for flexible data visualization and forecasting.
        </p>
    </div>
    <hr style="border: 1px solid #ccc;">
    """, unsafe_allow_html=True
)

# How to Use the App
st.markdown(
    """
    <div style="text-align: center;">
        <h3>🛠️ How to Use the App 🛠️</h3>
        <ol style="text-align: left; display: inline-block; margin: auto; padding-left: 15px;">
            <li>📌 Navigate to the **sidebar** on the left.</li>
            <li>📥 Enter the **stock ticker symbol** (e.g., AAPL for Apple, TSLA for Tesla).</li>
            <li>⏱️ Select the **time period** and **data interval** for the stock data.</li>
            <li>📈 View the historical data and predictions on the main page.</li>
        </ol>
    </div>
    <hr style="border: 1px solid #ccc;">
    """, unsafe_allow_html=True
)

# Key Features
st.markdown(
    """
    <div style="text-align: center;">
        <h3>✨ Key Features ✨</h3>
        <ul style="list-style-type: none;">
            <li>📊 **Interactive Charts**: Visualize stock trends with candlestick plots.</li>
            <li>🤖 **Machine Learning Predictions**: See future stock price.</li>
            <li>🌈 **User-Friendly Interface**: Easy to navigate and understand.</li>
        </ul>
    </div>
    <hr style="border: 1px solid #ccc;">
    """, unsafe_allow_html=True
)

# Disclaimer
st.markdown(
    """
    <div style="text-align: center; color: red;">
        <h3>⚠️ Disclaimer ⚠️</h3>
        <p>
        - The predictions provided by this app are for informational purposes only.  
        - Please consult with a financial advisor for investment decisions.
        </p>
    </div>
    <hr style="border: 1px solid #ccc;">
    """, unsafe_allow_html=True
)

# Footer
st.markdown(
    """
    <div style="text-align: center;">
        <p>🚀 Developed by <b>[Amanpreet Kaur]</b>. 🚀</p>
        <p>🌐 Empowering data-driven investment decisions! 🌐</p>
    </div>
    """, unsafe_allow_html=True
)
