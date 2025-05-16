# Algorithmic-Trading-Model
Algo Trading Model for NIFTY 50 Prediction
Project Overview
This repository contains a machine learning-based algorithmic trading project focused on predicting the movement of the NIFTY 50 index. The project implements multiple models—XGBoost, Random Forest, and LSTM—to classify whether the stock price will go up or down on a given day (Label = 1 if Close > Open, else 0). It also includes a regression-based XGBoost model for predicting closing prices and interactive dashboards for visualization.
The primary goal is to develop and compare machine learning models for stock price prediction, leveraging historical NIFTY 50 data and technical indicators like SMA, EMA, RSI, and MACD. The project is ideal for those interested in algorithmic trading, financial data analysis, and machine learning applications in finance.
Dataset
The dataset used is NIFTY 50_Historical_PR_01042010to01042025.csv, containing historical price data for the NIFTY 50 index from April 1, 2010, to April 1, 2025. Key columns include:

Date: Date of the trading day (e.g., "01 Apr 2025").
Open: Opening price of the index.
High: Highest price during the day.
Low: Lowest price during the day.
Close: Closing price of the index.

An additional dataset, NIFTY 50_Historical_PR_01042015to26042025.csv, is used for some cells, covering a slightly different date range (April 1, 2015, to April 26, 2025).
Features

Classification Models:
XGBoost: Achieves 73.02% accuracy (Cell 1) and 75.64% with technical indicators (Cell 2).
Random Forest: Achieves ~74–76% accuracy with technical indicators.
LSTM: Achieves ~77–78% accuracy with technical indicators, leveraging temporal patterns.


Regression Model:
XGBoost for predicting closing prices, with hyperparameter tuning and evaluation metrics (RMSE, R², MAPE).


Technical Indicators:
Simple Moving Average (SMA)
Exponential Moving Average (EMA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)


Interactive Dashboards:
Dash app for visualizing actual vs. predicted closing prices, residuals, feature importance, and correlations (Cell 3).
Plotly dashboard with candlestick charts, SMA/EMA, RSI, and buy/sell signals (Cell 4).



Requirements
To run this project, install the required Python packages:
pip install pandas numpy scikit-learn xgboost tensorflow plotly dash

Python Version
The notebook uses Python 3.12.7. Ensure your environment matches this version for compatibility.
Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/algo-trading-nifty50.git
cd algo-trading-nifty50


Install Dependencies:Run the command above to install the required packages.

Download the Dataset:

Place NIFTY 50_Historical_PR_01042010to01042025.csv and NIFTY 50_Historical_PR_01042015to26042025.csv in the project directory.
Alternatively, update the file paths in the notebook to match your local setup.


Run the Notebook:Open Models.ipynb in Jupyter Notebook or JupyterLab and execute the cells sequentially:
jupyter notebook Models.ipynb



Usage
Classification Models

Cell 1 (XGBoost): Trains a basic XGBoost classifier using Open, High, and Low features. Outputs accuracy (73.02%), confusion matrix, and classification report.
Cell 2 (XGBoost with Indicators): Enhances the XGBoost model by adding technical indicators (SMA, EMA, RSI, MACD), improving accuracy to 75.64%.
Random Forest and LSTM: Implemented in conversation history (not in Models.ipynb). These models achieve competitive accuracies (~74–76% for Random Forest, ~77–78% for LSTM with indicators).

Regression Model

Cell 3 (XGBoost Regression with Dash):
Trains an XGBoost regression model to predict closing prices.
Uses hyperparameter tuning with GridSearchCV.
Visualizes results via an interactive Dash app, showing actual vs. predicted prices, residuals, feature importance, and correlations.



Visualization

Cell 4 (Interactive Plotly Dashboard):
Displays a candlestick chart with SMA and EMA overlays.
Includes RSI with buy/sell signals (buy when RSI < 30, sell when RSI > 70).
Features a range slider and selector for zooming into specific time periods.



Results

Classification Performance:
XGBoost (Basic): 73.02% accuracy.
XGBoost (with Indicators): 75.64% accuracy.
Random Forest: ~74–76% accuracy.
LSTM: ~77–78% accuracy (best performance, leveraging temporal patterns).


Regression Performance:
Metrics (Cell 3) include RMSE, R², and MAPE, displayed in the Dash app.


Insights:
Technical indicators significantly improve model performance.
LSTM outperforms non-temporal models due to its ability to capture sequential patterns in stock data.



Future Improvements

Feature Engineering: Add more technical indicators (e.g., Bollinger Bands, Stochastic Oscillator) or external factors (e.g., market sentiment, macroeconomic data).
Model Optimization: Experiment with deeper LSTM architectures or ensemble methods.
Real-Time Deployment: Integrate with FPGAs for faster inference in live trading scenarios.
Backtesting: Implement a backtesting framework to evaluate trading strategies based on model predictions.

Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, open an issue first to discuss the proposed updates.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, reach out via GitHub Issues or email at [your-email@example.com].
