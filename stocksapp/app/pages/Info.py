import streamlit as st
import pandas as pd
from helper import*
from datetime import datetime, timedelta


# Streamlit page configuration
st.set_page_config(
    page_title="Stock Info",
    page_icon="🏛️",
)
#####Sidebar Start#####

# Add a sidebar
st.sidebar.markdown("## **User Input Features**")

# Fetch and store the stock data
stock_dict = fetch_stocks_label()

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", stock_dict)

# Add a selector for stock exchange
st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# Build the stock ticker
stock_ticker = f"{stock}"

# Add a disabled input for stock ticker
st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(
    label="Stock ticker code", placeholder=stock_ticker, disabled=True
)

#####Sidebar End#####


# Fetch the info of the stock
try:
    stock_data_info = fetch_stocks(stock_ticker)
except:
    st.error("Error: Unable to fetch the stock data. Please try again later.")
    st.stop()


##### Title #####

# Add title to the app
st.markdown("# **Stock Info Plus**")

# Add a subtitle to the app
st.markdown("##### **Enhancing Your Stock Market Insights**")

##### Basic Information #####

# Add a heading
st.markdown("## **Basic Information**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Issuer Name": [stock_data_info["Company Details"]["name"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Symbol": [stock_ticker]}),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame({"Currency": ['USD']}),  # Adjust currency if applicable
    hide_index=True,
    width=500,
)
col2.dataframe(pd.DataFrame({"Exchange": [stock_exchange]}), hide_index=True, width=500)

##### Basic Information End #####

##### Market Data #####

# Add a heading
st.markdown("## **Market Data**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Current Price": [stock_data_info["Stock Details"]["currentPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Previous Close": [stock_data_info["Stock Details"]["previousClose"]]}),
    hide_index=True,
    width=500,
)

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Stock Details"]["open"]]}),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Day Low": [stock_data_info["Stock Details"]["dayLow"]]}),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame({"Day High": [stock_data_info["Stock Details"]["dayHigh"]]}),
    hide_index=True,
    width=300,
)

##### Market Data End #####

##### Volume and Shares #####

# Add a heading
st.markdown("## **Volume and Shares**")

# Create 2 columns
col1, col2 = st.columns(2)

col1.dataframe(
    pd.DataFrame({"Volume": [stock_data_info["Volume and Shares"]["volume"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Regular Market Volume": [stock_data_info["Volume and Shares"]["regularMarketVolume"]]}
    ),
    hide_index=True,
    width=500,
)

##### Volume and Shares End #####

##### Dividends and Yield #####

# Add a heading
st.markdown("## **Dividends and Yield**")

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame(
        {"Dividend Rate": [stock_data_info["Dividends and Yield"]["dividendRate"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {"Dividend Yield": [stock_data_info["Dividends and Yield"]["dividendYield"]]}
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Payout Ratio": [stock_data_info["Dividends and Yield"]["payoutRatio"]]}
    ),
    hide_index=True,
    width=300,
)

##### Dividends and Yield End #####

##### Cash Flow #####

# Add a heading
st.markdown("## **Cash Flow**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 2
col1.dataframe(
    pd.DataFrame({"Free Cash Flow": [stock_data_info["Cash Flow"]["freeCashflow"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Operating Cash Flow": [stock_data_info["Cash Flow"]["operatingCashflow"]]}
    ),
    hide_index=True,
    width=500,
)

##### Cash Flow End #####

##### Analyst Targets #####

# Add a heading
st.markdown("## **Analyst Targets**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Target High Price": [stock_data_info["Analyst Targets"]["targetHighPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Target Low Price": [stock_data_info["Analyst Targets"]["targetLowPrice"]]}),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame({"Target Mean Price": [stock_data_info["Analyst Targets"]["targetMeanPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Target Median Price": [stock_data_info["Analyst Targets"]["targetMedianPrice"]]}
    ),
    hide_index=True,
    width=500,
)

# Row 3
col1.dataframe(
    pd.DataFrame(
        {"Recommendation Key": [stock_data_info["Analyst Targets"]["recommendationKey"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Recommendation Mean": [stock_data_info["Analyst Targets"]["recommendationMean"]]}
    ),
    hide_index=True,
    width=500,
)

##### Analyst Targets End #####

##### Governance and Risks #####

# Add a heading
st.markdown("## **Governance and Risks**")

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame({"Audit Risk": [stock_data_info["Governance and Risks"]["auditRisk"]]}),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Board Risk": [stock_data_info["Governance and Risks"]["boardRisk"]]}),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame({"Compensation Risk": [stock_data_info["Governance and Risks"]["compensationRisk"]]}),
    hide_index=True,
    width=300,
)

# Row 2
col1.dataframe(
    pd.DataFrame(
        {"Shareholder Rights Risk": [stock_data_info["Governance and Risks"]["shareHolderRightsRisk"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Overall Risk": [stock_data_info["Governance and Risks"]["overallRisk"]]}),
    hide_index=True,
    width=300,
)

##### Governance and Risks End #####
