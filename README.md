# Algorithmic-Trading-Project

Description of function I used in code

calculate_metrics:
  Function to compute statistics of the trading session for each asset. As input I take information about trades which were make in period of   time. As output I return win rate, average gain/loss ratio and sharpe ratio

calculate_momentum_signal:
  Purpose of the function is to calculate sell and buy signal based on sell_windiw, buy_window and threshold. I am using this function
  multiple times inside optimize_global_paramiters to find best parameters

simulate_trading:
  Role of this function is to simulate trading for test data. Input is test data and initial capital. Output is history of all trades. Output
  of this function serve as an input to calculate_metrics and to draw charts

optimize_global_parameters:
  The funtion is commented because running it is the most time consuming. It calculate best buy_window, sell_window and threshold. 
  Evaluation method is based on total profit on all trades. After calculating it once I simply used results in next attempts, because 
  results will be the same because initial data are the same

Main code:
  In main part of code I am downloading data using yfinance library, then using paramiters I am simulating trading for test data for each asset separatly. And finally I am drawing a charts.

