import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Function to calculate Fibonacci levels
def calculate_fibonacci_levels(high, low):
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

# Function to search for a value within a tolerance
def search_value_in_columns(data, value, tolerance, cols_to_search):
    condition = None
    
    for col in cols_to_search:
        if pd.api.types.is_numeric_dtype(data[col]):  # If column is numeric
            col_condition = (data[col] >= value - tolerance) & (data[col] <= value + tolerance)
        else:
            continue  # Skip non-numeric columns

        condition = col_condition if condition is None else condition | col_condition  # Combine conditions

    return data[condition] if condition is not None else pd.DataFrame()

# Function to calculate RSI manually
def calculate_rsi(data, period=14):
    delta = data.diff()  # Calculate daily price changes
    gain = delta.where(delta > 0, 0)  # Gains (only positive changes)
    loss = -delta.where(delta < 0, 0)  # Losses (only negative changes, converted to positive)

    # Calculate the exponential moving average of gains and losses
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Handle the case where avg_loss is zero to avoid division by zero
    rs = avg_gain / avg_loss
    rs = rs.replace({0: 0.0001})  # Replace 0 with a very small number

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Streamlit app
def main():
    st.set_page_config(page_title="Fibonacci & RSI Analyzer", layout="wide")
    st.title("Fibonacci Levels Calculator with RSI and EMAs")
    st.write("Fetch stock data and calculate Fibonacci retracement and extension levels along with RSI and EMAs.")

    # Sidebar for user inputs
    st.sidebar.header("Data Selection")
    
    # Input ticker and fetch data
    ticker = st.sidebar.text_input("Enter the stock ticker (e.g., AAPL, ^GDAXI for DAX):", value="^GDAXI")
    period = st.sidebar.selectbox("Select data period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.sidebar.selectbox("Select data interval", options=["1d", "1wk", "1mo"], index=0)

    st.sidebar.header("Search Options")
    # Search Type Selection
    search_type = "Value Search"  # Fixed to "Value Search"

    fib_levels = None  # Initialize fib_levels

    if ticker:
        try:
            # Fetch data
            stock_data = yf.download(ticker, period=period, interval=interval)

            if stock_data.empty:
                st.error("No data available for the given ticker.")
                return

            # Reset index and clean data
            stock_data.reset_index(inplace=True)
            stock_data['Date'] = stock_data['Date'].dt.date  # Remove timestamp from date

            # Calculate percentage change for Close
            stock_data['% Difference'] = stock_data['Close'].pct_change() * 100

            # Calculate the difference from the previous day's Close
            stock_data['Price Difference'] = (stock_data['Close'] - stock_data['Close'].shift(1)).round(2)

            # Calculate Volume Difference in millions (absolute volume difference in millions)
            stock_data['Volume Difference'] = (stock_data['Volume'].diff() / 1e6).round(2)  # Difference in millions
            stock_data['% Volume Difference'] = stock_data['Volume'].pct_change() * 100
            stock_data['% Volume Difference'] = stock_data['% Volume Difference'].round(2)

            # Calculate RSI manually
            stock_data['RSI'] = calculate_rsi(stock_data['Close'], period=14).round(2)

            # Calculate EMAs
            stock_data['EMA5'] = stock_data['Close'].ewm(span=5, adjust=False).mean().round(2)
            stock_data['EMA14'] = stock_data['Close'].ewm(span=14, adjust=False).mean().round(2)
            stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean().round(2)

            # Format all numeric columns to two decimal places where appropriate
            for col in stock_data.select_dtypes(include=['float', 'int']).columns:
                stock_data[col] = stock_data[col].round(2)

            # Sidebar for Search Options (Value Search)
            st.sidebar.subheader("Value Search Parameters")
            # Enter value and tolerance
            value_to_search = st.sidebar.number_input("Enter the value to search for (e.g., daily Close):", value=0.0)
            tolerance = st.sidebar.slider(
                "Select the tolerance:",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0
            )

            # Columns to search in
            available_cols = [col for col in stock_data.columns if pd.api.types.is_numeric_dtype(stock_data[col])]
            cols_to_search = st.sidebar.multiselect(
                "Select columns to search in",
                options=available_cols,
                default=available_cols
            )

            if not cols_to_search:
                st.sidebar.error("Please select at least one column to search.")
                st.stop()

            # Search for matches
            result = search_value_in_columns(stock_data, value_to_search, tolerance, cols_to_search)
            st.subheader("Value Search Results")
            if result.empty:
                st.write("No matches found.")
            else:
                # Display the search result dataframe with dynamic width
                st.dataframe(result, use_container_width=True)

                # Select a row for Fibonacci calculation
                row_index = st.selectbox("Select the row index for Fibonacci calculation", options=result.index)
                selected_row = result.loc[row_index]

                # Extract scalar values for high and low prices
                high_price = selected_row['High']
                low_price = selected_row['Low']

                if isinstance(high_price, pd.Series):
                    high_price = high_price.iloc[0]

                if isinstance(low_price, pd.Series):
                    low_price = low_price.iloc[0]

                # Calculate Fibonacci levels
                fib_levels = calculate_fibonacci_levels(high_price, low_price)

                # Format Fibonacci levels to two decimal places
                formatted_fib_levels = {key: round(value, 2) for key, value in fib_levels.items()}
                fib_df = pd.DataFrame(formatted_fib_levels.items(), columns=["Level", "Price"])

                # Display Fibonacci levels dataframe with dynamic width
                st.subheader("Fibonacci Levels")
                st.write("**Note:** Golden ratio 61.8%")
                st.dataframe(fib_df, use_container_width=True)

                # Option to download search results
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Search Results as CSV",
                    data=csv,
                    file_name='value_search_results.csv',
                    mime='text/csv',
                )

            # Plotting the stock data with Fibonacci levels and EMAs
            fig = go.Figure()

            # Plot stock's closing price as Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_data['Date'],
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Candlestick'
            ))

            # Add EMAs
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['EMA5'],
                mode='lines',
                name='EMA5',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['EMA14'],
                mode='lines',
                name='EMA14',
                line=dict(color='orange', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['EMA26'],
                mode='lines',
                name='EMA26',
                line=dict(color='green', width=1)
            ))

            # Add Fibonacci lines (only if fib_levels is defined)
            if fib_levels:
                for level, price in fib_levels.items():
                    fig.add_hline(y=price, line_dash="dash", line_color="red",
                                 annotation_text=level, annotation_position="top right")

            # Update main figure layout
            fig.update_layout(
                title=f'{ticker} Fibonacci Levels with EMAs',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white',
                height=600
            )

            # Add RSI subplot
            fig_rsi = go.Figure()

            fig_rsi.add_trace(go.Scatter(
                x=stock_data['Date'],
                y=stock_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))

            # Add RSI Overbought and Oversold lines
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red",
                             annotation_text="Overbought", annotation_position="top left")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green",
                             annotation_text="Oversold", annotation_position="bottom left")

            fig_rsi.update_layout(
                title=f'{ticker} Relative Strength Index (RSI)',
                xaxis_title='Date',
                yaxis_title='RSI',
                template='plotly_white',
                height=300
            )

            # Display the main chart and RSI chart
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig_rsi, use_container_width=True)

        except ValueError as ve:
            st.error(f"Invalid value entered: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
