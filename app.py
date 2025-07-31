import streamlit as st
import pandas as pd
from prophet import Prophet
import os

st.set_page_config(page_title="StockVision", layout="centered")
st.title("📈 StockVision: Costco & Amazon Forecast")

ticker = st.selectbox("Choose a stock", ["COST", "AMZN"])
csv_file = f"{ticker}.csv"

if not os.path.exists(csv_file):
    st.error(f"No CSV file found for {ticker}")
else:
    df = pd.read_csv(csv_file)
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.rename(columns={'ds': 'Date'})

    st.subheader("📊 7-Day Forecast")
    st.line_chart(forecast.set_index('Date')[['yhat']])
    st.dataframe(forecast.tail(7).set_index('Date').style.format("{:.2f}"))
