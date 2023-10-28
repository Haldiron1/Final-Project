# Final-Project

# Stock Dashboard Web App with Streamlit

This Python script creates a web application for visualizing and analyzing stock data. The app is built using Streamlit, a popular web app framework, and it provides various functionalities for exploring historical data, technical indicators, financial metrics, and making forecasts.

## Features:

1. **Insert Your Ticker:**
   - Allows users to input a stock ticker symbol (e.g., AAPL) to retrieve data.

2. **Historical Data:**
   - Displays historical stock data, including Open, High, Low, Close prices, and Daily Returns.
   - Provides basic statistics such as Mean Daily Return, Standard Deviation of Daily Return, and Annualized Sharpe Ratio.

3. **Technicals and Financials:**
   - Shows stock performance with Close Price and Simple Moving Averages (20, 50, and 100-day).
   - Provides Balance Sheet, Quarterly Balance Sheet, Cashflow, Quarterly Cashflow, Income Statement, and Quarterly Income Statement.

4. **Technical Indicators:**
   - Calculates and displays technical indicators like Average Directional Movement Index (ADX), Aroon Indicator, and Mass Index.
   - Plots ADX, Aroon Indicator, and Mass Index.

5. **Additional Metrics:**
   - Computes various trading metrics such as Gain/Pain Ratio, Payoff Ratio, Profit Factor, Common Sense Ratio, CPC Index, Tail Ratio, Outlier Win Ratio, and Outlier Loss Ratio.

6. **Forecasting:**
   - Allows users to predict stock prices using the Prophet forecasting model.
   - Displays forecasted data, prediction intervals, and components.

## How to Use:

1. Clone the repository or download the script.
2. Install the required libraries (Streamlit, yfinance, prophet, plotly, yahoo_fin, pandas, numpy, matplotlib) using `pip install -r requirements.txt`.
3. Run the script using `streamlit run FINALPROJECT.py`.
4. Input the stock ticker symbol and select the desired tab for analysis.
