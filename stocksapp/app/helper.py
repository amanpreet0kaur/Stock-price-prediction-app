import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings
import yfinance as yf  # For stock data
import pandas as pd  # For data handling
import numpy as np  # For numerical computations
from sklearn.preprocessing import MinMaxScaler  # For normalization and denormalization
from tensorflow.keras.models import Sequential, load_model  # For model creation and loading
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def fetch_stocks_label():
    # Example stock tickers
    stock_list = ["AAPL", "GOOGL", "MSFT"]
    return stock_list


def fetch_periods_intervals():
    # Define periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    return periods

def fetch_stocks(stock_ticker):
    stock_data=yf.Ticker(stock_ticker)
    stock_data_info = stock_data.info
    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")
    # Extract full stock data
    stock_data_info = stock_data.info

    # Extract only the important information
    stock_data_info = {
        'Company Details': {
            'name': stock_data_info['longName'],
            'symbol': stock_data_info['symbol'],
            'industry': stock_data_info['industry'],
            'sector': stock_data_info['sector'],
            'address': stock_data_info['address1'],
            'city': stock_data_info['city'],
            'state': stock_data_info['state'],
            'zip': stock_data_info['zip'],
            'country': stock_data_info['country'],
            'phone': stock_data_info['phone'],
            'website': stock_data_info['website'],
            'longBusinessSummary': stock_data_info['longBusinessSummary'],
            'fullTimeEmployees': stock_data_info['fullTimeEmployees'],
            'companyOfficers': stock_data_info['companyOfficers']
        },
        'Stock Details': {
            'currentPrice': stock_data_info['currentPrice'],
            'previousClose': stock_data_info['previousClose'],
            'open': stock_data_info['open'],
            'dayLow': stock_data_info['dayLow'],
            'dayHigh': stock_data_info['dayHigh'],
            'regularMarketPreviousClose': stock_data_info['regularMarketPreviousClose'],
            'regularMarketOpen': stock_data_info['regularMarketOpen'],
            'regularMarketDayLow': stock_data_info['regularMarketDayLow'],
            'regularMarketDayHigh': stock_data_info['regularMarketDayHigh'],
            'fiftyTwoWeekLow': stock_data_info['fiftyTwoWeekLow'],
            'fiftyTwoWeekHigh': stock_data_info['fiftyTwoWeekHigh'],
            'fiftyDayAverage': stock_data_info['fiftyDayAverage'],
            'twoHundredDayAverage': stock_data_info['twoHundredDayAverage'],
            'marketCap': stock_data_info['marketCap']
        },
        'Volume and Shares': {
            'volume': stock_data_info['volume'],
            'regularMarketVolume': stock_data_info['regularMarketVolume'],
            'averageVolume': stock_data_info['averageVolume'],
            'averageVolume10days': stock_data_info['averageVolume10days'],
            'sharesOutstanding': stock_data_info['sharesOutstanding'],
            'floatShares': stock_data_info['floatShares'],
            'impliedSharesOutstanding': stock_data_info['impliedSharesOutstanding']
        },
        'Dividends and Yield': {
            'dividendRate': stock_data_info['dividendRate'],
            'dividendYield': stock_data_info['dividendYield'],
            'exDividendDate': stock_data_info['exDividendDate'],
            'payoutRatio': stock_data_info['payoutRatio'],
            'trailingAnnualDividendRate': stock_data_info['trailingAnnualDividendRate'],
            'trailingAnnualDividendYield': stock_data_info['trailingAnnualDividendYield']
        },
        'Financial Performance': {
            'totalRevenue': stock_data_info['totalRevenue'],
            'revenuePerShare': stock_data_info['revenuePerShare'],
            'earningsGrowth': stock_data_info['earningsGrowth'],
            'revenueGrowth': stock_data_info['revenueGrowth'],
            'returnOnAssets': stock_data_info['returnOnAssets'],
            'returnOnEquity': stock_data_info['returnOnEquity'],
            'grossMargins': stock_data_info['grossMargins'],
            'operatingMargins': stock_data_info['operatingMargins']
        },
        'Financial Ratios': {
            'priceToBook': stock_data_info['priceToBook'],
            'debtToEquity': stock_data_info['debtToEquity'],
            'enterpriseToRevenue': stock_data_info['enterpriseToRevenue'],
            'enterpriseToEbitda': stock_data_info['enterpriseToEbitda'],
            'beta': stock_data_info['beta'],
            'trailingPE': stock_data_info['trailingPE'],
            'forwardPE': stock_data_info['forwardPE']
        },
        'Cash Flow': {
            'totalCash': stock_data_info['totalCash'],
            'totalCashPerShare': stock_data_info['totalCashPerShare'],
            'freeCashflow': stock_data_info['freeCashflow'],
            'operatingCashflow': stock_data_info['operatingCashflow']
        },
        'Analyst Targets': {
            'targetHighPrice': stock_data_info['targetHighPrice'],
            'targetLowPrice': stock_data_info['targetLowPrice'],
            'targetMeanPrice': stock_data_info['targetMeanPrice'],
            'targetMedianPrice': stock_data_info['targetMedianPrice'],
            'recommendationKey': stock_data_info['recommendationKey'],
            'recommendationMean': stock_data_info['recommendationMean']
        },
        'Governance and Risks': {
            'auditRisk': stock_data_info['auditRisk'],
            'boardRisk': stock_data_info['boardRisk'],
            'compensationRisk': stock_data_info['compensationRisk'],
            'shareHolderRightsRisk': stock_data_info['shareHolderRightsRisk'],
            'overallRisk': stock_data_info['overallRisk']
        }
    }
    return stock_data_info


def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]
    return stock_data_history


def predict_next_day(model, data, scaler, time_step=30):
    """
    Predict the next day's stock price using the trained model.
    """
    # Use the last `time_step` days
    last_data = data[-time_step:]
    last_data = last_data.reshape(1, time_step, 1)  # Reshape for model input

    # Predict the next day
    next_day_prediction = model.predict(last_data)

    # Denormalize the prediction
    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    return next_day_prediction.flatten()[0]


def generate_stock_prediction(stock_ticker):
    try:








        # Fetch stock data
        stock_data = yf.Ticker(stock_ticker)
        stock_data_hist = stock_data.history(period="10y", interval="1d")

        # Preprocess data
        stock_data_hist = stock_data_hist.reset_index()
        df = stock_data_hist[['Date', 'Close']]
        df1 = df.reset_index()['Close']

        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

        # Split data
        train_size = int(len(df1) * 0.7)
        val_size = int(len(df1) * 0.1)
        train_data = df1[:train_size]
        validation_data = df1[train_size:train_size + val_size]
        test_data = df1[train_size + val_size:]

        # Create datasets
        def create_dataset(dataset, time_step=1):
            datax, datay = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                datax.append(a)
                datay.append(dataset[i + time_step, 0])
            return np.array(datax), np.array(datay)

        time_step = 30
        x_train, y_train = create_dataset(train_data, time_step)
        x_test, y_test = create_dataset(test_data, time_step)
        x_val, y_val = create_dataset(validation_data, time_step)

        # Reshape data for LSTM
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)




        #creating dates


        df2=df.reset_index()['Date']
        df2=np.array(df2).reshape(-1,1)
        train_size_date = int(len(df2) * 0.8)  # 70% for training
        val_size_date = int(len(df2) * 0.1)    # 10% for validation
        test_size_date = len(df2) - train_size - val_size  # Remaining 20% for testing
        train_data_date = df2[0:train_size, :]
        validation_data_date = df2[train_size:train_size + val_size, :]
        test_data_date = df2[train_size + val_size:, :1]

        time_step=30
        x_train_date,y_train_date=create_dataset(train_data_date,time_step)
        x_test_date,y_test_date=create_dataset(test_data_date,time_step)
        x_val_date,y_val_date=create_dataset(validation_data_date,time_step)

        # Define LSTM model
        model = Sequential([
            layers.Input((30, 1)),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error']
        )

        # Callbacks
        checkpoint = ModelCheckpoint(
            filepath='best_lstm_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train the model
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=50,
            callbacks=[checkpoint, early_stop]
        )

        # Load the best saved model
        model = load_model('best_lstm_model.keras')

        # Predictions
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        val_pred = model.predict(x_val)

        # Denormalize predictions
        train_pred = scaler.inverse_transform(train_pred)
        test_pred = scaler.inverse_transform(test_pred)
        val_pred = scaler.inverse_transform(val_pred)

        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

        # Create DataFrames for results
        train_df = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': train_pred.flatten()})
        test_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': test_pred.flatten()})
        val_df = pd.DataFrame({'Actual': y_val.flatten(), 'Predicted': val_pred.flatten()})

        # Predict next day price
        next_day_price = predict_next_day(model, df1, scaler)

        return train_df, test_df, val_df, next_day_price,y_train_date,y_test_date,y_val_date

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None


