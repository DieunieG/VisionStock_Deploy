import streamlit as st
import pandas as pd
from prophet import Prophet
import os

st.set_page_config(page_title="StockVision", layout="centered")
st.title("📈 StockVision: Costco & Amazon Forecast")

ticker = st.selectbox("Choose a stock", ["COST", "AMZN"])
csv_file = f"{ticker}.csv"

if not os.path.exists(csv_file):
    st.error(f"No CSV file found for {ticker}. Please ensure the file exists.")
else:
    try:
        # Load and preprocess the data
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'Date' not in df.columns or 'Close' not in df.columns:
            st.error("CSV file must contain 'Date' and 'Close' columns.")
        else:
            df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # Convert 'ds' to datetime with error handling
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            
            # Check for invalid dates
            if df['ds'].isnull().any():
                invalid_dates = df[df['ds'].isnull()]['ds'].index.tolist()
                st.error(f"Invalid dates found in CSV at rows: {invalid_dates}. Please ensure all dates are in a valid format (e.g., YYYY-MM-DD).")
            else:
                # Ensure 'y' (Close) is numeric
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                if df['y'].isnull().any():
                    st.error("Some 'Close' values are invalid or non-numeric. Please check the CSV file.")
                else:
                    # Fit Prophet model
                    model = Prophet()
                    model.fit(df)

                    # Generate 7-day forecast
                    future = model.make_future_dataframe(periods=7)
                    forecast = model.predict(future)

                    # Select relevant columns and rename
                    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    forecast = forecast.rename(columns={'ds': 'Date'})

                    # Display forecast
                    st.subheader("📊 7-Day Forecast")
                    
                    # Create line chart for the forecast
                    st.line_chart(forecast.set_index('Date')[['yhat']].rename(columns={'yhat': 'Forecasted Close'}))

                    # Display the forecast table
                    st.dataframe(
                        forecast.tail(7)[['Date', 'yhat', 'yhat_lower', 'yhat_upper']]
                        .set_index('Date')
                        .style.format({"yhat": "{:.2f}", "yhat_lower": "{:.2f}", "yhat_upper": "{:.2f}"})
                    )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please check the CSV file format and contents.")