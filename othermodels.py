import datetime
import numpy as np
import pandas as pd
import sklearn
import pickle

from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
import datetime
import numpy as np
import pandas as pd
import sklearn

from pandas.io.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from backtest import Strategy, Portfolio
from sklearn.qda import QDA

import matplotlib.pyplot as plt


class MarketIntradayPortfolio(Portfolio):
    """Buys or sells 500 shares of an asset at the opening price of
    every bar, depending upon the direction of the forecast, closing 
    out the trade at the close of the bar.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0, shares=500):
        self.symbol = symbol        
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()
        
    def generate_positions(self):
        """Generate the positions DataFrame, based on the signals
        provided by the 'signals' DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)

        positions[self.symbol] = self.shares*self.signals['signal']
        return positions
                    
    def backtest_portfolio(self):
        """Backtest the portfolio and return a DataFrame containing
        the equity curve and the percentage returns."""
       
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()
            
        portfolio['price_diff'] = self.bars['Close']-self.bars['Open']
        portfolio['price_diff'][0:5] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']
     
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the 
    adjusted closing value of a stock obtained from Yahoo Finance, along with 
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = DataReader(symbol, "yahoo", start_date-datetime.timedelta(days=365), end_date)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Create the shifted lag series of prior trading period close values
    for i in xrange(0,lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in xrange(0,lags):
        tsret["Lag%s" % str(i+1)] = tslag["Lag%s" % str(i+1)].pct_change()*100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print "%s: %.3f" % (name, hit_rate)

if __name__ == "__main__":
    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series("^DJI", datetime.datetime(2008,8,8), datetime.datetime(2016,07,01), lags=10)

    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2015,1,1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test
    symbol = "^DJI"
    # Create and fit the three models    
    print "Hit Rates:"
    models = [("LR", LogisticRegression())]


    predDict = pickle.load( open( "pickle/prediction.p", "rb" ))

    for m in models:
        f, ax = plt.subplots(1, sharex=True)
        f.patch.set_facecolor('white')

        fit_model(m[0], m[1], X_train, y_train, X_test, pred)
    	signals = X_test.copy()
    	
        signals['signal'] = predDict['KNeighborsClassifier 30']
        signals.signal[signals.signal == 0] = -1
        signals['positions'] = signals['signal'].diff()
        amount_of_shares = 100
        bars =  pd.io.data.get_data_yahoo("^DJI", start_test, datetime.datetime(2016,07,01))
        portfolio = MarketIntradayPortfolio("^DJI", bars, signals, initial_capital = 100000.0, shares = amount_of_shares)
        returns = portfolio.backtest_portfolio()
        returns['total'].plot(ax = ax, color='g', lw=3.)

        print "KNN Final " + str(returns.iloc[377]['total'])



        signals['signal'] = pred['LR']
    	signals.signal[signals.signal == 0] = -1
    	signals['positions'] = signals['signal'].diff()
    	amount_of_shares = 100
    	
    	portfolio = MarketIntradayPortfolio("^DJI", bars, signals, initial_capital = 100000.0, shares = amount_of_shares)
    	returns = portfolio.backtest_portfolio()
        print "Log Regression on Lags " + str(returns.iloc[377]['total'])
    	
    	ylabel = symbol + ' Close Price in $'
    	bars['Open'] =  bars['Open'] * 5.5
        bars['Open'].plot(ax=ax, color='r', lw=3.)
        print  'Buy and hold ' +  str(bars.iloc[377]['Open'])
        returns['total'].plot(ax=ax, color='b', lw=3.)
    	ax.set_ylabel('Portfolio value in $', fontsize=18)
    	ax.set_xlabel('Date', fontsize=18)
    	#ax[1].legend(('Portofolio Performance',), loc='upper left', prop={"size":18})
    	plt.tick_params(axis='both', which='major', labelsize=14)
    	loc = ax.xaxis.get_major_locator()
    	#loc.maxticks[DAILY] = 24
    	figManager = plt.get_current_fig_manager()
    	#figManager.window.showMaximized()
    	plt.savefig("backtesting.png")


