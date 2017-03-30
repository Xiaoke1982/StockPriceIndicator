# Investment and Trading Project: Build a Stock Price Indicator
## Machine Learning: Regression

## Overview
This project is a partial requirement of Machine Learning Engineer Nano Degree at Udacity. This is the Capstone Project of the Nano Degree. This project consists of two main tasks. The first main task is to build a stock price predictor that takes daily Adjusted Close price data of certain stocks over a certain date range and a prediction interval as input, and outputs the prediction of future Adjusted Close price of the stocks for a given query date. The stock price predictor is built as a user-friendly interface such that users can specify stock(s) they are interested in, the training date range, and the prediction interval. To achieve this, several regression models are fitted to predict the rate of change of the stocks based on the technical indicators generated from the stocks' daily adjusted close prices. Then, these models are evaluated and the optimal one is selected for future predictions. The regression methods used in this project include K-Nearest-Neighbor, Bagging, Decision Tree and Random Forest. For each method, different values of tunning parameter(s) are attempted. The evaluation metrics I used in the project is the mean absolute percentage error (MAPE). 

The second main task is to use the stock price predictor we built to construct a trading strategy that suggests when to buy or sell a certain stock. Both of the stock price predictor and the trading strategy will be tested and evaluated in 3 different case studies. The data we use in this project are queried directly from Yahoo! Finance. There are 3 case studies in this project. The first study uses daily trading  data of Google. The Google’s daily Adjusted Close prices ranging from 2011
to 2014 will be used to train a stock price predictor. Then a trading strategy that suggests when to buy or sell Google stock is constructed based on the predictor. The performance of both the predictor and the trading strategy will be tested using the daily Google stock data of the year 2015. In the second study, we use the daily trading data of Apple ranging from 2003 to 2007 as training data, and use the daily trading data of Apple in year 2008 as testing data to conduct a study in the same way as we do in the first study. Thirdly, the same study is conducted using the IBM daily stock data ranging from 2010 to 2013 as training data, and the IBM daily stock data in 2014 as testing data. In all the 3 studies, we will attempt 3 different prediction intervals: 5, 10 and 20 trading days respectively. The reason that we conduct 3 case studies is that we want to investigate if our stock price predictor and corresponding trading strategy can perform consistently well in different cases.

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

The project also contains three other files:

1. 

## Run the Smart Cab Agent  

In a terminal, navigate to the top-level project directory smartcab/ and run the following commands:

- python smartcab/agent.py


## Method and Result

This project applies the Q-learning algorithm to train a self-driving agent such that it can arrives its destination in the allotted time. First of all, The state of the agent in the Q-table I defined consists of 2 parts. The first part is the current traffic light at current intersection. The second part is the direction that the agent should drive leading to the destination at current intersection. After the state is defined, a Q-table is updated using the Q-learning algorithm during the training phase. The learning rate in Q-learning algorithm decays over time as 1/t. For the discount factor, 9 different values were attempted, and the optimal one was selected. In order to avoid "local minimum", the epsilon-method is applied. The epsilon also decays over time. Many different values of epsilon are attempted in a systematic way, and the optimal one was selected. 

In the project, the Q-table is trained and updated for 90 trials. Then the agent is tested for the following 10 trails to see in how many trails the agent can successfully arrive the destination within the given time. In order to get a more stable result, this process repeats 10 times and an averaged number of successful trails out of 10 is computed. This averaged number is used as the evaluation metrics for both tunning and testing. 

The results showed that the optimal discount factor is 0.4. Also, the optimal epsilon should decay 0.00046 at each time step. Using the optimal combination of discount factor and epsilon, the agent can arrive the destination within reasonable time in 8 out of 10 trails on average. When checking the Q-table we trained, it can be found that 5 out of 6 actions chosen based on the Q-values match the corresponding correct actions. The details can be found in the report file. 
