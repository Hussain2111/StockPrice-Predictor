import yfinance as yf # imports package to download sp500 data
import pandas as pd # imports package to manipulate data

'''
Access the price history and query it to show all the history since the start of the sp500 index.
'''
sp500 = yf.Ticker("^GSPC") 
sp500 = sp500.history(period="max")
sp500
sp500.index

'''
Plot the line graph for the closing price.
'''
sp500.plot.line(y="Close", use_index=True)

del sp500["Dividends"] # delete the column of dividends
del sp500["Stock Splits"] # delete the column of the stock splits

'''
Create a column named Tomorrow where the value of each row is the value of the closing price of the previous column
'''
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500

'''
Create a column named "Target" where it returns '1' for true if today's price is greater than yesterday's price or return '0' for false instead.
'''
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500

'''
Remove the rows which are of the period before '1990'
'''
sp500 = sp500.loc["1990-01-01":].copy()
sp500

'''
Machine learning model initialization
'''
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)






