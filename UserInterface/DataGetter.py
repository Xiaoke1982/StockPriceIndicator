import pandas_datareader.data as web
import pandas as pd 
import numpy as np 
import datetime as dt


def get_data(symbols, train_sd, data_source = "yahoo"):

    """
    The function takes some symbols and a starting date as input, and extracts 
        close prices data of these symbols and SPY from yahoo finance website, 
        ranging from the specified starting date to present.

	@param symbols:  a list, symbols to extract from yahoo finance
	@param train_sd: a datetime object, training data start date  
	@return:         a dataframe containing adjusted close prices of the symbols 
                     and SPY from train_sd to present
    """
    if "SPY" not in symbols:
        symbols = symbols + ["SPY"]

    df = web.DataReader(name=symbols[0], data_source=data_source, start=train_sd)[["Adj Close"]]
    #use double [[]] returns a DataFrame, not a Series
    df = df.rename(columns={"Adj Close": symbols[0]})

    for i in range(1, len(symbols)):
    	df_temp = web.DataReader(name=symbols[i], data_source=data_source, start=train_sd)[["Adj Close"]]
    	df_temp = df_temp.rename(columns={"Adj Close": symbols[i]})
    	df = df.join(df_temp)

    df = df.dropna(subset=["SPY"])

    return df


def process_data(df_allprices, symbol, train_ed, n_days):
    """
    The function computes technical indicators based on the prices data obtained by get_data() function, 
        and returns training set and testing set of a specified symbol.
    The returned training and testing set are dataframes that contain technical indicators as columns.
    The returned dataframes also contains rate of change as target variable, stock prices and future actual stock prices.
    The indicators contains: Billinger Bonds values for 5, 10 and 20 trading days of the symbol's prices
                             Billinger Bonds values for 5, 10 and 20 trading days of the SPY prices
                             Rate of Change values for the past 5, 10 and 20 trading days of the symbol's prices
                             Rate of Change values for the past 5, 10 and 20 trading days of the SPY prices
                             RSI values for 5, 10 and 20 trading days of the symbol's prices
                             RSI values for 5, 10 and 20 trading days of the SPY prices
    The input n_days denotes the prediction interval, 
        e.g. n_days=5 means you want to predict the stock price 5 trading days later.

	@param df_allprices: dataframe, adj close of all symbols and SPY 
	@param symbol:       str, the symbol to process 
	@param train_ed:     datetime object, training end date 
    @param n_days:       int, prediction interval, e.g. n_days=5 means you want to 
                         predict the stock price 5 trading days later.
	return: two dataframes with stock indicators of symbol and SPY
			Dataframes also contains rate of change as target variable (Y).
            Dataframes also contains stock prices and future stock prices.
			Two dataframs are training and testing sets respectively.
    """
    ans_df = pd.DataFrame(index=df_allprices.index)
    df_symbol = df_allprices[symbol]
    df_SPY = df_allprices["SPY"]
    
    sma_5 = pd.Series.rolling(df_symbol, window=5, center=False).mean()
    std_5 = pd.Series.rolling(df_symbol, window=5, center=False).std()
    bb_5 = (df_symbol - sma_5) / (2 * std_5)
    ans_df["BB_5"] = bb_5
    
    sma_10 = pd.Series.rolling(df_symbol, window=10, center=False).mean()
    std_10 = pd.Series.rolling(df_symbol, window=10, center=False).std()
    bb_10 = (df_symbol - sma_10) / (2 * std_10)
    ans_df["BB_10"] = bb_10
    
    sma_20 = pd.Series.rolling(df_symbol, window=20, center=False).mean()
    std_20 = pd.Series.rolling(df_symbol, window=20, center=False).std()
    bb_20 = (df_symbol - sma_20) / (2 * std_20)
    ans_df["BB_20"] = bb_20
    
    
    sma_5_SPY = pd.Series.rolling(df_SPY, window=5, center=False).mean()
    std_5_SPY = pd.Series.rolling(df_SPY, window=5, center=False).std()
    bb_5_SPY = (df_SPY - sma_5_SPY) / (2 * std_5_SPY)
    ans_df["BB_5_SPY"] = bb_5_SPY
    
    sma_10_SPY= pd.Series.rolling(df_SPY, window=10, center=False).mean()
    std_10_SPY = pd.Series.rolling(df_SPY, window=10, center=False).std()
    bb_10_SPY = (df_SPY - sma_10_SPY) / (2 * std_10_SPY)
    ans_df["BB_10_SPY"] = bb_10_SPY
    
    sma_20_SPY= pd.Series.rolling(df_SPY, window=20, center=False).mean()
    std_20_SPY = pd.Series.rolling(df_SPY, window=20, center=False).std()
    bb_20_SPY = (df_SPY - sma_20_SPY) / (2 * std_20_SPY)
    ans_df["BB_20_SPY"] = bb_20_SPY
    
    roc_5 = df_symbol[5:]/df_symbol[:-5].values - 1
    ans_df["ROC_5"] = roc_5
    
    roc_10 = df_symbol[10:]/df_symbol[:-10].values - 1
    ans_df["ROC_10"] = roc_10
    
    roc_20 = df_symbol[20:]/df_symbol[:-20].values - 1
    ans_df["ROC_20"] = roc_20
    
    roc_5_SPY = df_SPY[5:]/df_SPY[:-5].values - 1
    ans_df["ROC_5_SPY"] = roc_5_SPY
    
    roc_10_SPY = df_SPY[10:]/df_SPY[:-10].values - 1
    ans_df["ROC_10_SPY"] = roc_10_SPY
    
    roc_20_SPY = df_SPY[20:]/df_SPY[:-20].values - 1
    ans_df["ROC_20_SPY"] = roc_20_SPY
    
    delta = df_symbol.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    up_5 = pd.Series.rolling(up, window=5, center=False).mean()
    down_5 = pd.Series.rolling(down.abs(), window=5, center=False).mean()
    RS_5 = up_5 / down_5
    RSI_5 = 100.0 - (100.0 / (1.0 + RS_5))
    ans_df["RSI_5"] = RSI_5
    
    up_10 = pd.Series.rolling(up, window=10, center=False).mean()
    down_10 = pd.Series.rolling(down.abs(), window=10, center=False).mean()
    RS_10 = up_10 / down_10
    RSI_10 = 100.0 - (100.0 / (1.0 + RS_10))
    ans_df["RSI_10"] = RSI_10
    
    up_20 = pd.Series.rolling(up, window=20, center=False).mean()
    down_20 = pd.Series.rolling(down.abs(), window=20, center=False).mean()
    RS_20 = up_20 / down_20
    RSI_20 = 100.0 - (100.0 / (1.0 + RS_20))
    ans_df["RSI_20"] = RSI_20
    
    delta_SPY = df_SPY.diff()
    up_SPY, down_SPY = delta_SPY.copy(), delta_SPY.copy()
    up_SPY[up_SPY < 0] = 0
    down_SPY[down_SPY > 0] = 0
    
    up_5_SPY = pd.Series.rolling(up_SPY, window=5, center=False).mean()
    down_5_SPY = pd.Series.rolling(down_SPY.abs(), window=5, center=False).mean()
    RS_5_SPY = up_5_SPY / down_5_SPY
    RSI_5_SPY = 100.0 - (100.0 / (1.0 + RS_5_SPY))
    ans_df["RSI_5_SPY"] = RSI_5_SPY
    
    up_10_SPY = pd.Series.rolling(up_SPY, window=10, center=False).mean()
    down_10_SPY = pd.Series.rolling(down_SPY.abs(), window=10, center=False).mean()
    RS_10_SPY = up_10_SPY / down_10_SPY
    RSI_10_SPY = 100.0 - (100.0 / (1.0 + RS_10_SPY))
    ans_df["RSI_10_SPY"] = RSI_10_SPY
    
    up_20_SPY = pd.Series.rolling(up_SPY, window=20, center=False).mean()
    down_20_SPY = pd.Series.rolling(down_SPY.abs(), window=20, center=False).mean()
    RS_20_SPY = up_20_SPY / down_20_SPY
    RSI_20_SPY = 100.0 - (100.0 / (1.0 + RS_20_SPY))
    ans_df["RSI_20_SPY"] = RSI_20_SPY

    y = df_symbol[n_days:].values/df_symbol[:-n_days] - 1
    ans_df["Y"] = y

    ans_df["price"] = df_symbol
    ans_df["future_price"] = np.concatenate((df_symbol[n_days:].values, [np.nan]*n_days))
    

    train_df = ans_df[:train_ed]
    test_df = ans_df[train_ed+dt.timedelta(days=1):]
    train_df = train_df.dropna() #drop na from the training set
    
    return train_df, test_df

