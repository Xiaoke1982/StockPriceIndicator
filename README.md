# Investment and Trading Project: Build a Stock Price Indicator
## Machine Learning: Regression

## Overview
This project is a partial requirement of Machine Learning Engineer Nano Degree at Udacity. This is the Capstone Project of the Nano Degree. This project consists of two main tasks. The first main task is to build a stock price predictor that takes daily Adjusted Close price data of certain stocks over a certain date range and a prediction interval as input, and outputs the prediction of future Adjusted Close price of the stocks for a given query date. The stock price predictor is built as a user-friendly interface such that users can specify stock(s) they are interested in, the training date range, and the prediction interval. To achieve this, several regression models are fitted to predict the rate of change of the stocks based on the technical indicators generated from the stocks' daily adjusted close prices. Then, these models are evaluated and the optimal one is selected for future predictions. The regression methods used in this project include **K-Nearest-Neighbor, Bagging, Decision Tree** and **Random Forest**. For each method, different values of tunning parameter(s) are attempted. The evaluation metrics I used in the project is the mean absolute percentage error (MAPE). 

The second main task is to use the stock price predictor we built to construct a trading strategy that suggests when to buy or sell a certain stock. Both of the stock price predictor and the trading strategy will be tested and evaluated in 3 different case studies. The data we use in this project are queried directly from Yahoo! Finance. There are 3 case studies in this project. The first study uses daily trading  data of Google. The Google’s daily Adjusted Close prices ranging from 2011
to 2014 will be used to train a stock price predictor. Then a trading strategy that suggests when to buy or sell Google stock is constructed based on the predictor. The performance of both the predictor and the trading strategy will be tested using the daily Google stock data of the year 2015. In the second study, we use the daily trading data of Apple ranging from 2003 to 2007 as training data, and use the daily trading data of Apple in year 2008 as testing data to conduct a study in the same way as we do in the first study. Thirdly, the same study is conducted using the IBM daily stock data ranging from 2010 to 2013 as training data, and the IBM daily stock data in 2014 as testing data. In all the 3 studies, we will attempt 3 different prediction intervals: 5, 10 and 20 trading days respectively. The reason that we conduct 3 case studies is that we want to investigate if our stock price predictor and corresponding trading strategy can perform consistently well in different cases.  

The results showed that our stock price predictor performs consistently well on these three different testing sets. In both Google and IBM studies, the MAPEs of the predictor on testing sets are both below 3%, which means that the predicted stock value 5 days out is within +/- 3% of actual value on average. In the study of Apple, the MAPE on testing set is about 6%, which is just a slightly above 5%. The predictor can successfully predict the trend in all the three case studies.
In all the three cases, the trading strategy based on the predictions by our stock price predictor overcomes the stock market in terms of cumulative return. Using our trading strategy can earn more profit or lose less money than just buy the stock and simply hold it for all the 3 cases. Also, our strategy has less risk than the just holding the stock in terms of higher Sharpe ratio in both of the Google and IBM studies. In summary, we have shown that our stock price predictor and corresponding trading strategy have consistently good performances on different testing sets.

## Software Requirements

This project uses the following software and Python libraries:

1. Python 2.7
2. numpy
3. pandas
4. pandas_datareader
5. datetime
6. sklearn
7. matplotlib

## Files and Codes
This project contains one directory:

1. UserInterface: This folder contains the python scripts that create a user friendly interface of the stock price predictor.

In the directory UserInterface:

1. DataGetter.py: functions that extract data from Yahoo! Finance and pre-process data
2. KNN.py: OOP class for KNN model
3. BagLearner.py: OOP class for Bagging model
4. BestLeaner.py: OOP class for selecting best model among KNN, Bagging, Decision Tree and Random Forest
5. StockIndicator.py: Implementation for User Interface

The project also contains two other files:

1. StockProject.ipynb: ipython notebook that contains all the code, explanation and results of the project
2. StockProjectReport.pdf: the report of this project that illustrate the models, the structure of the project and the results in great detail. 

## Run the User Interface of Stock Price Predictor

In a terminal, navigate to the project directory UserInterface/ and run the following commands:

- python StockIndicator.py

Then input the information according to the prompt. 

