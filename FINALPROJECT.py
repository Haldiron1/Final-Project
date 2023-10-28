import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Title of your app
st.title("Stock Dashboard")
st.sidebar.success("After choosing the ticker, choose one of the Tabs below: Historical Data, Technicals and Financials or Forecasting")

# Initialize ticker variable
ticker = None

# Sidebar menu
with st.sidebar:
    selected = st.radio("Select a Tab", ["Insert Your Ticker", "Historical Data", "Technicals and Financials", "Forecasting"])

    # User Input for Ticker Symbol
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):")

    # User Input for Start and End Dates
    start_date = st.date_input("Start Date", value=None, min_value=None, max_value=None, key=None, help=None)
    end_date = st.date_input("End Date", value=None, min_value=None, max_value=None, key=None, help=None)

    if ticker and start_date and end_date:
        # Convert dates to string format
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        # Get Data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date).reset_index()
        # Reverse the DataFrame
        df = df.iloc[::-1]

        # Calculate daily returns
        if len(df) > 1:  # Check if there is enough data for calculation
            df['Daily Returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        else:
            df['Daily Returns'] = None

        # Invert the signs of daily_returns
        df['Daily Returns'] = df['Daily Returns'] * -1
        df['Daily Returns'] = df['Daily Returns'].shift(-1)

        # Add '%' symbol to the daily_returns column
        df['Daily Returns'] = df['Daily Returns'].apply(lambda x: f'{x:.2f}%')

        # Rename the column
        df = df.rename(columns={'Daily Returns': 'Daily Returns'})

        # Convert 'Daily Returns' column to numeric
        df['Daily Returns'] = df['Daily Returns'].str.rstrip('%').astype('float')

if selected == "Insert Your Ticker":
    pass


elif selected == "Historical Data":
    if ticker and start_date and end_date:
        # Display Data
        st.write(f"## Historical Data for {ticker}")
        st.write(df)

        # Display basic statistics
        st.write("## Basic Statistics")
        st.write(f"**Mean Daily Return:** {df['Daily Returns'].mean()}%")
        st.write("\n\n")
        st.write("\n\n")

        st.write("The \"Mean Daily Return\" refers to the average percentage change in the stock price over a given period. It indicates, on average, how much the stock's price tends to increase or decrease on a daily basis.",
                 "For example, if the mean daily return is 0.5%, it suggests that, on average, the stock's price increases by 0.5% each day. Conversely, if the mean daily return is -0.5%, it indicates that, on average, the stock's price decreases by 0.5% each day."
                )
        st.write("\n\n")
        st.write("\n\n")
        st.write(f"**Standard Deviation of Daily Return:** {df['Daily Returns'].std()}%")
        st.write("\n\n")
        st.write("\n\n")
        st.write(f"The \"Standard Deviation of Daily Return\" is a statistical measure that quantifies the amount of variation or dispersion in a set of data points. In the context of this stock dashboard, it specifically refers to the variability or spread of the daily returns of the selected stock.",
                 f"A higher standard deviation indicates that the daily returns are more spread out from the mean, suggesting higher volatility. Conversely, a lower standard deviation suggests that the daily returns are closer to the mean, indicating lower volatility.",
                 f"In the provided value, it means that, on average, the daily returns deviate by approximately {df['Daily Returns'].std()}% from the mean daily return." 
                )
        st.write("\n\n")
        st.write("\n\n")
        st.write(f"**Annualized Sharpe Ratio:** {np.sqrt(252) * df['Daily Returns'].mean() / df['Daily Returns'].std()}")
        st.write("\n\n")
        st.write("\n\n")
        st.write(f"The Annualized Sharpe Ratio is a measure of the risk-adjusted return of an investment or portfolio. It was developed by William F. Sharpe and is widely used to evaluate the performance of investments."

        f"In your case, the value '{np.sqrt(252) * df['Daily Returns'].mean() / df['Daily Returns'].std():.2f}' represents the Annualized Sharpe Ratio for the selected stock."

        "Here's what it means:"

        "- A Sharpe Ratio greater than 1 is generally considered good. It indicates that the investment is providing a higher return for the level of risk taken."

        "- A Sharpe Ratio less than 1 suggests that the investment is not providing sufficient return for the level of risk taken."

        "- A higher Sharpe Ratio indicates better risk-adjusted performance."

        "Keep in mind that a higher Sharpe Ratio is generally desirable, but it's important to consider other factors and perform a comprehensive analysis when evaluating investments."
    )
         #plot raw data with slider using plotly
    def plot_close_prices():
        print(df.columns)  # Add this line to debug
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
        fig.layout.update(title_text='Close Prices', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_close_prices()

elif selected == "Technicals and Financials":
    if ticker and start_date and end_date:
        st.write("## Stock Performance")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Close'], label='Close Price')

        # Calculate 20, 50, and 100-day SMAs
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA100'] = df['Close'].rolling(window=100).mean()

        # Plot SMAs
        ax.plot(df['SMA20'], label='20-day SMA', linestyle='--')
        ax.plot(df['SMA50'], label='50-day SMA', linestyle='--')
        ax.plot(df['SMA100'], label='100-day SMA', linestyle='--')

        ax.set_title(f"{ticker} Stock Price with SMAs")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        ticker_info = yf.Ticker(ticker)

        # Show balance sheet
        balance_sheet = ticker_info.balance_sheet
        st.write(f"## Balance Sheet for {ticker}")
        st.write(balance_sheet)

        quarterly_balance_sheet = ticker_info.quarterly_balance_sheet
        st.write(f"## Quarterly Balance Sheet for {ticker}")
        st.write(quarterly_balance_sheet)

        # Show cashflow
        cashflow = ticker_info.cashflow
        st.write(f"## Cashflow for {ticker}")
        st.write(cashflow)

        quarterly_cashflow = ticker_info.quarterly_cashflow
        st.write(f"## Quarterly Cashflow for {ticker}")
        st.write(quarterly_cashflow)

        # Show financials
        financials = ticker_info.financials
        st.write(f"## Income Statement for {ticker}")
        st.write(financials)

        quarterly_financials = ticker_info.quarterly_financials
        st.write(f"## Quarterly Income Statement for {ticker}")
        st.write(quarterly_financials)

        # Calculate Average Directional Movement Index (ADX)
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate True Range (TR)
        df['TR'] = np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))

        # Calculate Directional Movement (DM) and Directional Index (DI)
        df['DMplus'] = (high - high.shift(1)).clip(lower=0)
        df['DMminus'] = (low.shift(1) - low).clip(lower=0)

        # Calculate True Range, DMplus, and DMminus smoothed over a period (usually 14)
        period = 14
        df['TR_smooth'] = df['TR'].rolling(window=period).mean()
        df['DMplus_smooth'] = df['DMplus'].rolling(window=period).mean()
        df['DMminus_smooth'] = df['DMminus'].rolling(window=period).mean()

        # Calculate Directional Index (DI)
        df['DIplus'] = (df['DMplus_smooth'] / df['TR_smooth']) * 100
        df['DIminus'] = (df['DMminus_smooth'] / df['TR_smooth']) * 100

        # Calculate Average Directional Index (ADX)
        df['DX'] = (np.abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])) * 100
        df['ADX'] = df['DX'].rolling(window=period).mean()

        # Calculate Aroon Indicator
        def aroon(df, period=25):
            df['Aroon_Up'] = (df['High'].rolling(window=period).apply(lambda x: x.argmax(), raw=True) / period) * 100
            df['Aroon_Down'] = (df['Low'].rolling(window=period).apply(lambda x: x.argmin(), raw=True) / period) * 100

        aroon(df)

        # Calculate Mass Index (MI)
        def mass_index(df, period1=9, period2=25):
            df['High-Low'] = df['High'] - df['Low']
            df['High-Close'] = np.abs(df['High'] - df['Close'].shift(1))
            df['Low-Close'] = np.abs(df['Low'] - df['Close'].shift(1))

            df['Single EMA1'] = df['High-Close'].ewm(span=period1, min_periods=period1).mean() / df['Low-Close'].ewm(span=period1, min_periods=period1).mean()
            df['Double EMA1'] = df['Single EMA1'].ewm(span=period1, min_periods=period1).mean()
            df['Single EMA2'] = df['High-Close'].ewm(span=period2, min_periods=period2).mean() / df['Low-Close'].ewm(span=period2, min_periods=period2).mean()
            df['Double EMA2'] = df['Single EMA2'].ewm(span=period2, min_periods=period2).mean()

            df['Mass Index'] = df['Double EMA1'] / df['Double EMA2']

        mass_index(df)

        # Calculate Max Drawdown
        cumulative_returns = (1 + df['Daily Returns']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

            # Display the DataFrame with calculated indicators
        st.write(f"## Technical Indicators for {ticker}")
        st.write(df[['ADX', 'Aroon_Up', 'Aroon_Down', 'Mass Index']])

        # Calculate and plot Average Directional Movement Index (ADX)
        fig, ax_adx = plt.subplots(figsize=(10, 5))
        ax_adx.plot(df['ADX'], label='ADX')
        ax_adx.set_title(f"{ticker} Average Directional Movement Index (ADX)")
        ax_adx.set_xlabel("Date")
        ax_adx.set_ylabel("Value")
        ax_adx.legend()
        st.pyplot(fig)

        # Calculate and plot Aroon Indicator
        fig, ax_aroon = plt.subplots(figsize=(10, 5))
        ax_aroon.plot(df['Aroon_Up'], label='Aroon Up', color='green')
        ax_aroon.plot(df['Aroon_Down'], label='Aroon Down', color='red')
        ax_aroon.set_title(f"{ticker} Aroon Indicator")
        ax_aroon.set_xlabel("Date")
        ax_aroon.set_ylabel("Value")
        ax_aroon.legend()
        st.pyplot(fig)

        # Calculate and plot Mass Index (MI)
        fig, ax_mi = plt.subplots(figsize=(10, 5))
        ax_mi.plot(df['Mass Index'], label='Mass Index', color='purple')
        ax_mi.set_title(f"{ticker} Mass Index (MI)")
        ax_mi.set_xlabel("Date")
        ax_mi.set_ylabel("Value")
        ax_mi.legend()
        st.pyplot(fig)

        # Calculate the additional metrics
        def calculate_additional_metrics(data):
            # Calculate Gain/Pain Ratio
            positive_returns = data['Daily Returns'][data['Daily Returns'] > 0].sum()
            negative_returns = data['Daily Returns'][data['Daily Returns'] < 0].sum()
            gain_pain_ratio = positive_returns / abs(negative_returns)

            # Calculate Payoff Ratio
            average_gain = positive_returns / len(data[data['Daily Returns'] > 0])
            average_loss = abs(negative_returns) / len(data[data['Daily Returns'] < 0])
            payoff_ratio = average_gain / average_loss

            # Calculate Profit Factor
            profit_factor = positive_returns / abs(negative_returns)

            # Calculate Common Sense Ratio
            common_sense_ratio = positive_returns / max_drawdown

            # Calculate CPC Index
            cpc_index = (positive_returns - abs(negative_returns)) / (positive_returns + abs(negative_returns))

            # Calculate Tail Ratio
            tail_ratio = positive_returns.std() / abs(negative_returns.std())

            # Calculate Outlier Win Ratio
            positive_outliers = data['Daily Returns'][data['Daily Returns'] > 2*data['Daily Returns'].std()]
            negative_outliers = data['Daily Returns'][data['Daily Returns'] < -2*data['Daily Returns'].std()]
            outlier_win_ratio = len(positive_outliers) / len(negative_outliers)

            # Calculate Outlier Loss Ratio
            outlier_loss_ratio = len(negative_outliers) / len(positive_outliers)

            return {
                'Gain/Pain Ratio': gain_pain_ratio,
                'Payoff Ratio': payoff_ratio,
                'Profit Factor': profit_factor,
                'Common Sense Ratio': common_sense_ratio,
                'CPC Index': cpc_index,
                'Tail Ratio': tail_ratio,
                'Outlier Win Ratio': outlier_win_ratio,
                'Outlier Loss Ratio': outlier_loss_ratio
            }

        additional_metrics = calculate_additional_metrics(df)

        # Display the additional metrics
        st.write(f"## Additional Metrics for {ticker}")
        st.write(additional_metrics)

elif selected == "Forecasting":
    if ticker and start_date and end_date:
        st.write(f"## Forecasting for {ticker}")
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365
        df_train = df[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())

        # Show Prediction Intervals
        st.write("Prediction Intervals:")
        st.write(f"Lower Bound (5%): {forecast['yhat_lower'].iloc[-1]:.2f}")
        st.write(f"Upper Bound (95%): {forecast['yhat_upper'].iloc[-1]:.2f}")

        st.write(f'Forecast plot for {n_years} years')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write("Forecast components")
        fig2 = m.plot_components(forecast)
        st.write(fig2)
