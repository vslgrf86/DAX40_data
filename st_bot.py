import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Indices and assets to track
indices = {
    "Dow Jones": "^DJI",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "VIX": "^VIX",
    "Nikkei 225": "^N225",
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "Shanghai Composite": "000001.SS"
}

# Set Streamlit page configuration
st.set_page_config(
    page_title="Fibonacci Levels Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "Closing Prices and Differences"])

# Utility Functions
def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement and extension levels."""
    retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    extension_ratios = [1.0, 1.236, 1.382, 1.5, 1.618, 1.786, 2.0, 2.618]

    retracement_levels = {
        f"Retracement {int(ratio * 100)}%": low + (high - low) * ratio
        for ratio in retracement_ratios
    }
    extension_levels = {
        f"Extension {int(ratio * 100)}%": high + (high - low) * (ratio - 1)
        for ratio in extension_ratios
    }
    return {**retracement_levels, **extension_levels}

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, period):
    """Calculate Exponential Moving Average (EMA)."""
    return data.ewm(span=period, adjust=False).mean()

def search_value_in_columns(data, value, tolerance, cols_to_search):
    """Search for a value within a specified tolerance in selected columns."""
    condition = None
    for col in cols_to_search:
        if pd.api.types.is_numeric_dtype(data[col]):
            col_condition = (data[col] >= value - tolerance) & (data[col] <= value + tolerance)
            condition = col_condition if condition is None else condition | col_condition
    return data[condition] if condition is not None else pd.DataFrame()

def fetch_closing_prices(symbols, period, interval):
    """Fetch closing prices and % differences for given symbols."""
    data = {}
    for name, ticker in symbols.items():
        try:
            # Fetch data for the specified period and interval
            df = yf.download(ticker, period=period, interval=interval)
            if len(df) > 1:  # Ensure at least two rows of data
                latest_close = df['Close'].iloc[-1]
                previous_close = df['Close'].iloc[-2]
                percent_diff = ((latest_close - previous_close) / previous_close) * 100
                data[name] = {"Close": latest_close, "% Difference": percent_diff}
            else:
                data[name] = {"Close": None, "% Difference": None}  # Handle insufficient data
        except Exception:
            data[name] = {"Close": None, "% Difference": None}  # Handle errors gracefully
    return pd.DataFrame.from_dict(data, orient="index")

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period, interval):
    """Fetch stock data from Yahoo Finance."""
    return yf.download(ticker, period=period, interval=interval)

# Main App Logic
def main():
    if page == "Main Page":
        st.title("Fibonacci Levels Calculator")
        st.write("""
            Analyze stock data and calculate Fibonacci retracement and extension levels.
            Use the inputs below to interact with data and explore results.
        """)

        # Sidebar Inputs
        st.sidebar.header("User Inputs")
        ticker = st.sidebar.text_input("Enter stock ticker:", value="^GDAXI")
        period = st.sidebar.selectbox("Select data period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)
        interval = st.sidebar.selectbox("Select data interval:", ["1d", "1wk", "1mo"], index=0)

        if ticker:
            try:
                stock_data = fetch_stock_data(ticker, period, interval)

                if stock_data.empty:
                    st.error("No data available for the given ticker.")
                    return

                # Data Preparation
                stock_data.reset_index(inplace=True)
                stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date  # Keep only the date
                stock_data.drop(columns=["Adj Close"], inplace=True)  # Drop "Adjusted Close" column

                stock_data['Price Difference'] = stock_data['Close'].diff()
                stock_data['% Price Difference'] = stock_data['Close'].pct_change() * 100
                stock_data['Volume Difference'] = stock_data['Volume'].diff() / 1e6
                stock_data['% Volume Difference'] = stock_data['Volume'].pct_change() * 100

                stock_data['RSI'] = calculate_rsi(stock_data['Close'])
                stock_data['EMA 5'] = calculate_ema(stock_data['Close'], 5)
                stock_data['EMA 14'] = calculate_ema(stock_data['Close'], 14)
                stock_data['EMA 26'] = calculate_ema(stock_data['Close'], 26)
                
                # EMA for Volume
                stock_data['EMA 5 (Vol)'] = calculate_ema(stock_data['Volume'], 5)
                stock_data['EMA 14 (Vol)'] = calculate_ema(stock_data['Volume'], 14)
                stock_data['EMA 26 (Vol)'] = calculate_ema(stock_data['Volume'], 26)

                stock_data['Year'] = pd.to_datetime(stock_data['Date']).dt.year.astype(int)
                stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.month
                stock_data['Quarter'] = pd.to_datetime(stock_data['Date']).dt.quarter
                stock_data['Calendar Week'] = pd.to_datetime(stock_data['Date']).dt.isocalendar().week
                stock_data['Week in Quarter'] = stock_data.groupby('Quarter')['Calendar Week'].transform(lambda x: x - x.min() + 1)
                stock_data['Weekday'] = pd.to_datetime(stock_data['Date']).dt.day_name()
            
                # Format percentage columns to include the '%' sign
                percentage_cols = ['% Price Difference', '% Volume Difference']
                for col in percentage_cols:
                    stock_data[col] = stock_data[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

                # Format numerical columns to two decimal places
                numerical_cols = stock_data.select_dtypes(include='number').columns
                stock_data[numerical_cols] = stock_data[numerical_cols].round(2)

                # Data Preview
                st.subheader("Data Preview")
                st.dataframe(stock_data, use_container_width=True)

                # Search Functionality - Central Inputs
                st.subheader("Search in Data")
                numeric_cols = stock_data.select_dtypes(include='number').columns.tolist()
                cols_to_search = st.multiselect("Columns to search:", ["All"] + numeric_cols, default=["All"])
                if "All" in cols_to_search:
                    cols_to_search = numeric_cols

                search_value = st.number_input("Search Value:", value=0.0)
                tolerance = st.slider("Tolerance:", 0.0, 50.0, 10.0, key="close_price_tolerance")

                result = search_value_in_columns(stock_data, search_value, tolerance, cols_to_search)
                st.subheader("Search Results")
                if result.empty:
                    st.warning("No matches found.")
                else:
                    st.dataframe(result, use_container_width=True)

                    # Fibonacci Calculation
                    selected_row = st.selectbox("Select row for Fibonacci:", result.index)
                    high, low = result.loc[selected_row, ['High', 'Low']]
                    fib_levels = calculate_fibonacci_levels(high, low)

                    st.subheader("Fibonacci Levels")
                    fib_df = pd.DataFrame(fib_levels.items(), columns=["Level", "Price"])
                    fib_df['Price'] = fib_df['Price'].round(2)  # Round prices to two decimals
                    st.dataframe(fib_df, use_container_width=True)   
                    
                    # Select Fibonacci Level for Searching
                    st.subheader("Search Fibonacci Levels in Data")
                    fib_level_to_search = st.selectbox("Select Fibonacci Level to Search:", fib_df['Price'])
                    
                    # Tolerance for Searching
                    tolerance = st.slider("Tolerance:", 0.0, 50.0, 10.0, key="fib_tolerance")  # Default ±10
                    
                    # Filter stock data for Fibonacci Levels
                    fib_search_results = stock_data[
                        (stock_data['High'] >= fib_level_to_search - tolerance) &
                        (stock_data['High'] <= fib_level_to_search + tolerance)
                    ]
                    
                    # Display Search Results
                    if not fib_search_results.empty:
                        st.write(f"### Matches for {fib_level_to_search} ± {tolerance}:")
                        st.dataframe(fib_search_results, use_container_width=True)
                    else:
                        st.warning(f"No matches found for {fib_level_to_search} ± {tolerance}.")

                    # Plot Chart with Fibonacci Levels
                    st.subheader("Candlestick Chart with Fibonacci Levels")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=stock_data['Date'],
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name="Candlestick"
                    ))

                    for level, price in fib_levels.items():
                        fig.add_hline(y=price, line_dash="dash", annotation_text=level, line_color="blue")

                    fig.update_layout(
                        title=f"{ticker} Fibonacci Levels",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_white",
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif page == "Closing Prices and Differences":
        # Fetch data
        st.subheader("Closing Prices and % Differences for Major Indices and Assets")
        data = fetch_closing_prices(indices, "1mo", "1d")  # Example values for period and interval
        data.reset_index(inplace=True)
        data.rename(columns={"index": "Asset"}, inplace=True)  # Rename the index column
        data["Close"] = data["Close"].round(2)  # Round closing prices to 2 decimals
        data["% Difference"] = data["% Difference"].round(2)  # Round % difference to 2 decimals
        st.dataframe(data, use_container_width=True)

if __name__ == "__main__":
    main()
