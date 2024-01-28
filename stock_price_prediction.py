mport streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import Ridge
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense

st.header("Stock market prediction")

st.markdown("##### Choose a company from the given list")

# List of top 200 stocks
top_200_stocks = [
    "Select company",
    'Reliance Industries Ltd.', 'Tata Consultancy Services Ltd.', 'HDFC Bank Ltd.', 'Infosys Ltd.', 'Hindustan Unilever Ltd.',
    'ICICI Bank Ltd.', 'Kotak Mahindra Bank Ltd.', 'Bajaj Finance Ltd.', 'Housing Development Finance Corporation Ltd.', 'State Bank of India',
    'Maruti Suzuki India Ltd.', 'Nestle India Ltd.', 'Axis Bank Ltd.', 'ITC Ltd.', 'Bharti Airtel Ltd.',
    'Asian Paints Ltd.', 'Mahindra & Mahindra Ltd.', 'UltraTech Cement Ltd.', 'Sun Pharmaceutical Industries Ltd.', 'Tech Mahindra Ltd.',
    'Power Grid Corporation of India Ltd.', 'Titan Company Ltd.', 'Bajaj Auto Ltd.', 'Dr. Reddy\'s Laboratories Ltd.', 'Wipro Ltd.',
    'NTPC Ltd.', 'Oil & Natural Gas Corporation Ltd.', 'Shree Cement Ltd.', 'Tata Steel Ltd.', 'Hindalco Industries Ltd.',
    'IndusInd Bank Ltd.', 'Larsen & Toubro Ltd.', 'Britannia Industries Ltd.', 'Grasim Industries Ltd.', 'Cipla Ltd.',
    'Adani Ports & Special Economic Zone Ltd.', 'JSW Steel Ltd.', 'HCL Technologies Ltd.', 'GAIL (India) Ltd.', 'Divi\'s Laboratories Ltd.',
    'Hero MotoCorp Ltd.', 'Eicher Motors Ltd.', 'Coal India Ltd.', 'Indian Oil Corporation Ltd.', 'SBI Life Insurance Company Ltd.',
    'Hindustan Petroleum Corporation Ltd.', 'Bharat Petroleum Corporation Ltd.', 'ICICI Prudential Life Insurance Company Ltd.', 'Tata Consumer Products Ltd.',
    'TCS',
    'RELIANCE.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'HINDUNILVR.NS',
    'ICICIBANK.NS',
    'KOTAKBANK.NS',
    'BAJFINANCE.NS',
    'HDFC.NS',
    'SBIN.NS',
    'MARUTI.NS',
    'NESTLEIND.NS',
    'AXISBANK.NS',
    'ITC.NS',
    'BHARTIARTL.NS',
    'ASIANPAINT.NS',
    'M&M.NS',
    'ULTRACEMCO.NS',
    'SUNPHARMA.NS',
    'TECHM.NS',
    'POWERGRID.NS',
    'TITAN.NS',
    'BAJAJ-AUTO.NS',
    'DRREDDY.NS',
    'WIPRO.NS',
    'NTPC.NS',
    'ONGC.NS',
    'SHREECEM.NS',
    'TATASTEEL.NS',
    'HINDALCO.NS',
    'INDUSINDBK.NS',
    'LT.NS',
    'BRITANNIA.NS',
    'GRASIM.NS',
    'CIPLA.NS',
    'ADANIPORTS.NS',
    'JSWSTEEL.NS',
    'HCLTECH.NS',
    'GAIL.NS',
    'DIVISLAB.NS',
    'HEROMOTOCO.NS',
    'EICHERMOT.NS',
    'COALINDIA.NS',
    'IOC.NS',
    'SBILIFE.NS',
    'HINDPETRO.NS',
    'BPCL.NS',
    'ICICIPRULI.NS',
    'TATACONSUM.NS'
]

selected_option = st.selectbox('Search for a company', top_200_stocks)

if selected_option != 'Select company':
    all_stocks = {
        "Reliance Industries Ltd.": "RELIANCE.NS",
        "Tata Consultancy Services Ltd.": "TCS.NS",
        "HDFC Bank Ltd.": "HDFCBANK.NS",
        "Infosys Ltd.": "INFY.NS",
        "Hindustan Unilever Ltd.": "HINDUNILVR.NS",
        "ICICI Bank Ltd.": "ICICIBANK.NS",
        "Kotak Mahindra Bank Ltd.": "KOTAKBANK.NS",
        "Bajaj Finance Ltd.": "BAJFINANCE.NS",
        "Housing Development Finance Corporation Ltd.": "HDFC.NS",
        "State Bank of India": "SBIN.NS",
        "Maruti Suzuki India Ltd.": "MARUTI.NS",
        "Nestle India Ltd.": "NESTLEIND.NS",
        "Axis Bank Ltd.": "AXISBANK.NS",
        "ITC Ltd.": "ITC.NS",
        "Bharti Airtel Ltd.": "BHARTIARTL.NS",
        "Asian Paints Ltd.": "ASIANPAINT.NS",
        "Mahindra & Mahindra Ltd.": "M&M.NS",
        "UltraTech Cement Ltd.": "ULTRACEMCO.NS",
        "Sun Pharmaceutical Industries Ltd.": "SUNPHARMA.NS",
        "Tech Mahindra Ltd.": "TECHM.NS",
        "Power Grid Corporation of India Ltd.": "POWERGRID.NS",
        "Titan Company Ltd.": "TITAN.NS",
        "Bajaj Auto Ltd.": "BAJAJ-AUTO.NS",
        "Dr. Reddy's Laboratories Ltd.": "DRREDDY.NS",
        "Wipro Ltd.": "WIPRO.NS",
        "NTPC Ltd.": "NTPC.NS",
        "Oil & Natural Gas Corporation Ltd.": "ONGC.NS",
        "Shree Cement Ltd.": "SHREECEM.NS",
        "Tata Steel Ltd.": "TATASTEEL.NS",
        "Hindalco Industries Ltd.": "HINDALCO.NS",
        "IndusInd Bank Ltd.": "INDUSINDBK.NS",
        "Larsen & Toubro Ltd.": "LT.NS",
        "Britannia Industries Ltd.": "BRITANNIA.NS",
        "Grasim Industries Ltd.": "GRASIM.NS",
        "Cipla Ltd.": "CIPLA.NS",
        "Adani Ports & Special Economic Zone Ltd.": "ADANIPORTS.NS",
        "JSW Steel Ltd.": "JSWSTEEL.NS",
        "HCL Technologies Ltd.": "HCLTECH.NS",
        "GAIL (India) Ltd.": "GAIL.NS",
        "Divi's Laboratories Ltd.": "DIVISLAB.NS",
        "Hero MotoCorp Ltd.": "HEROMOTOCO.NS",
        "Eicher Motors Ltd.": "EICHERMOT.NS",
        "Coal India Ltd.": "COALINDIA.NS",
        "Indian Oil Corporation Ltd.": "IOC.NS",
        "SBI Life Insurance Company Ltd.": "SBILIFE.NS",
        "Hindustan Petroleum Corporation Ltd.": "HINDPETRO.NS",
        "Bharat Petroleum Corporation Ltd.": "BPCL.NS",
        "ICICI Prudential Life Insurance Company Ltd.": "ICICIPRULI.NS",
        "Tata Consumer Products Ltd.": "TATACONSUM.NS"
    }
    
    ticker_symbol = all_stocks[selected_option]
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history("12mo")

    if not data.empty:
        train_x = data['Open']
        train_y = data['High']

        st.subheader("Actual vs Predicted Prices")

        model = keras.Sequential([
            Dense(units=25, activation="relu"),
            Dense(units=5, activation='relu'),
            Dense(units=1, activation='relu'),
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_x, train_y, epochs=125)

        predicted_prices = model.predict(train_x)

        actual_vs_predicted = pd.DataFrame({
            'Date': data.index,
            'Actual': train_y,
            'Predicted': predicted_prices.flatten()
        })

        st.line_chart(actual_vs_predicted.set_index('Date'))

        # Calculate threshold-based prediction
        threshold = 0.5  # Example threshold value
        predicted_price = predicted_prices[-1]
        current_price = data['Close'].iloc[-1]  # Assuming 'Close' price is used for prediction

        if predicted_price > current_price * (1 + threshold):
            st.markdown("### According to my knowledge, tomorrow the {0} stock price is predicted to increase.".format(selected_option))
        elif predicted_price < current_price * (1 - threshold):
            st.markdown("### According to my knowledge, tomorrow the {0} stock price is predicted to decrease.".format(selected_option))
        else:
            st.markdown("### According to my knowledge, tomorrow the {0} stock price is predicted to remain relatively stable.".format(selected_option))

    else:
        st.write("No data available for the selected company.")

# Buttons for Nifty 50 and Bank Nifty
if st.button('Show Nifty 50'):
    nifty_50_data = yf.download("^NSEI", start="2023-01-01", end="2024-01-01")
    st.subheader("Nifty 50")
    st.line_chart(nifty_50_data[['Adj Close', 'Open']])

if st.button('Show Bank Nifty'):
    bank_nifty_data = yf.download("^NSEBANK", start="2023-01-01", end="2024-01-01")
    st.subheader("Bank Nifty")
    st.line_chart(bank_nifty_data[['Adj Close', 'Open']])
