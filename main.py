import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Function to calculate momentum and generate buy/sell signals with dynamic parameters
def calculate_metrics(gains_losses, capital_history, num_trades):
    # Win rate
    num_winning_trades = sum(1 for gain in gains_losses if gain > 0)
    win_rate = num_winning_trades / num_trades if num_trades > 0 else 0.0

    # Average gain/loss ratio
    gains = [gain for gain in gains_losses if gain > 0]
    losses = [gain for gain in gains_losses if gain < 0]
    average_gain = np.mean(gains) if len(gains) > 0 else 0
    average_loss = np.mean(losses) if len(losses) > 0 else 0
    avg_gain_loss_ratio = average_gain / abs(average_loss) if average_loss != 0 else np.inf

    # Sharpe ratio (using daily returns)
    daily_returns = pd.Series(np.diff(capital_history) / capital_history[:-1])  # Daily returns
    avg_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0

    # Maximum drawdown
    peak = np.argmax(np.maximum.accumulate(capital_history))
    trough = np.argmin(capital_history[peak:])
    max_drawdown = (capital_history[peak + trough] - capital_history[peak]) / capital_history[peak] if peak < len(
        capital_history) and trough < len(capital_history) else 0

    return win_rate, avg_gain_loss_ratio, sharpe_ratio, max_drawdown

# Function to calculate momentum and generate buy/sell signals with dynamic parameters
def calculate_momentum_signals(data, buy_window, sell_window, threshold=0.0):
    data['Buy Momentum'] = data['Close'].pct_change(periods=buy_window)  # Buy momentum
    data['Sell Momentum'] = data['Close'].pct_change(periods=sell_window)  # Sell momentum

    # Generate buy signals: strong positive momentum
    data['Buy Signal'] = np.where(data['Buy Momentum'] > threshold, 1, 0)

    # Generate sell signals: weaker momentum or decline
    data['Sell Signal'] = np.where(data['Sell Momentum'] <= threshold, 1, 0)

    return data


# Function to simulate trading based on signals (with buy/sell enforcement)
def simulate_trading(data, initial_capital=1000):
    cash = initial_capital
    position = 0  # Position indicates whether you own stock or not
    capital_history = [initial_capital]  # Start capital history with the initial value
    trade_history = []  # To track buy/sell points
    gains_losses = []  # To track gains and losses for win/loss ratio
    num_trades = 0  # Counter for number of trades

    for i in range(1, len(data)):
        # Buy condition: Only buy if we don't already have a position
        if data['Buy Signal'].iloc[i] == 1 and position == 0:  # Buy condition
            position = cash / data['Close'].iloc[i]  # Buy the stock
            cash = 0  # All capital is now in the stock
            trade_history.append(('buy', data['Date'].iloc[i], data['Close'].iloc[i]))  # Track buy
            num_trades += 1  # Increment trade count

        # Sell condition: Only sell if we have a position
        elif data['Sell Signal'].iloc[i] == 1 and position > 0:  # Sell condition
            sell_price = data['Close'].iloc[i]
            cash = position * sell_price  # Sell the stock
            gains_losses.append(sell_price - data['Close'].iloc[i - 1])  # Record gain/loss for trade
            trade_history.append(('sell', data['Date'].iloc[i], sell_price))  # Track sell
            position = 0  # Reset position after selling

        # Track capital history at each step
        capital_history.append(cash + position * data['Close'].iloc[i] if position > 0 else cash)

    final_value = capital_history[-1] if capital_history else initial_capital
    return final_value, capital_history, trade_history, gains_losses, num_trades


# Main processing
symbols = ["PKO.WA", "EUR=X", "NQ=F", "CL=F", "TLT"]
start_date = "2008-01-01"
end_date = "2023-01-01"

all_data = []
for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date)
    data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    data = data.reset_index()
    all_data.append(data)

# Split data into train and test sets
train_data = {}
test_data = {}
for i, symbol in enumerate(symbols):
    data = all_data[i]
    # Training data: 2008-01-01 to 2018-12-31
    train_data[symbol] = data[data['Date'] <= '2018-12-31']
    # Testing data: 2019-01-01 to 2023-01-01
    test_data[symbol] = data[data['Date'] >= '2019-01-01']

# Optimized parameters set manually (as you mentioned)
# def optimize_global_parameters(train_data):
#     best_profit = -np.inf
#     best_params = None
#
#     # Parameter ranges for optimization
#     buy_windows = range(3, 15)  # Test buy window sizes
#     sell_windows = range(3, 15)  # Test sell window sizes
#     thresholds = [0.0, 0.005, 0.01, 0.02]  # Test momentum thresholds
#
#     for buy_window, sell_window, threshold in product(buy_windows, sell_windows, thresholds):
#         total_profit = 0
#
#         # Evaluate each parameter set across all instruments in the training data
#         for data in train_data:
#             temp_data = calculate_momentum_signals(data.copy(), buy_window, sell_window, threshold)
#             final_value, _, _ = simulate_trading(temp_data)
#             total_profit += final_value  # Accumulate total profit
#
#         if total_profit > best_profit:
#             best_profit = total_profit
#             best_params = (buy_window, sell_window, threshold)
#
#     return best_params

buy_window = 5
sell_window = 13
threshold = 0.0

# Apply the optimized parameters and evaluate for the test data
results = {}
for i, symbol in enumerate(symbols):
    data = test_data[symbol]  # Get the corresponding stock data from test_data
    data = calculate_momentum_signals(data, buy_window, sell_window, threshold)
    final_value, capital_history, trade_history, gains_losses, num_trades = simulate_trading(data)

    # Calculate the metrics
    win_rate, avg_gain_loss_ratio, sharpe_ratio, max_drawdown = calculate_metrics(gains_losses, capital_history,
                                                                                  num_trades)

    results[symbol] = {
        'Final Portfolio Value': final_value,
        'Win Rate': win_rate,
        'Average Gain/Loss Ratio': avg_gain_loss_ratio,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

    # Print results for each stock
    print(f"Results for {symbol}:")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Gain/Loss Ratio: {avg_gain_loss_ratio:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print("-" * 40)

    # Calculate the value of $1000 invested in the stock on 01.01.2019
    initial_investment = 1000
    stock_investment = initial_investment * (data['Close'] / data['Close'].iloc[0])

    # Plot the interactive chart for each stock's portfolio performance
    fig = go.Figure()

    # Plot portfolio value over time
    fig.add_trace(go.Scatter(x=data['Date'], y=capital_history, mode='lines', name=f'{symbol} Portfolio Value'))

    # Plot stock's close price over time for comparison
    fig.add_trace(go.Scatter(x=data['Date'], y=stock_investment, mode='lines', name=f'{symbol} Stock Investment', line=dict(dash='dash')))

    # Customize layout
    fig.update_layout(
        title=f"Portfolio vs Stock Investment for {symbol}",
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='closest',
        showlegend=True
    )

    # Update legend names
    legend_names = {
        "EUR=X": "USD/EUR",
        "NQ=F": "Nasdaq",
        "CL=F": "Crude Oil",
        "TLT": "20y US Bonds",
    }

    # Update legend labels in the traces
    for trace in fig.data:
        if trace.name in legend_names:
            trace.name = legend_names[trace.name]

    # Show interactive chart
    fig.show()

    # Create the candle chart for stock price with buy/sell points
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{symbol} Candlestick'
    )])

    # Buy Points: Detect where Buy Signal changes from 0 to 1
    buy_points = data[(data['Buy Signal'] == 1) & (data['Buy Signal'].shift(1) == 0)]

    # Sell Points: Detect where Sell Signal changes from 1 to 0
    sell_points = data[(data['Sell Signal'] == 0) & (data['Sell Signal'].shift(1) == 1)]

    # Track Buy Price and associated Sell Points
    sell_gains = []  # List to track whether each sell was a gain or loss
    buy_price = None  # Variable to hold the buy price for matching with a sell

    for i, sell_point in enumerate(sell_points['Date']):
        # Find the corresponding buy point from trade_history (it should be before the current sell point)
        buy_point = None
        for trade in reversed(trade_history):
            if trade[0] == 'buy' and trade[1] < sell_point:
                buy_point = trade
                break  # We found the most recent buy before the current sell point

        if buy_point:
            buy_price = buy_point[2]  # The price at which the stock was bought
            sell_price = sell_points.iloc[i]['Close']  # The price at which the stock was sold

            # Check if the sell was a gain or loss
            if sell_price > buy_price:
                sell_gains.append('green')  # Gain
            else:
                sell_gains.append('red')  # Loss

    # Now, plot the sell points with corresponding colors (green for gains, red for losses)
    for i, sell_point in enumerate(sell_points['Date']):
        color = sell_gains[i] if i < len(sell_gains) else 'red'  # Default to 'red' if no color is assigned
        fig_candle.add_trace(go.Scatter(
            x=[sell_point],
            y=[sell_points.iloc[i]['Close']],
            mode='markers',
            marker=dict(color=color, size=10, symbol='circle'),
            name=f"Sell Signal ({color.capitalize()})"
        ))

    # Customize layout for the candlestick chart
    fig_candle.update_layout(
        title=f"{symbol} Candlestick Chart with Sell Signals (Green for Gains, Red for Losses)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=True
    )

    # Show the candlestick chart
    fig_candle.show()

# If you want to see the results in a dataframe
results_df = pd.DataFrame(results).T  # Transpose for easier viewing
print("Overall Results:")
print(results_df)
