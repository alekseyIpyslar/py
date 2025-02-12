import time
import datetime as dt

import yfinance as yf
from fontTools.designspaceLib import posix

short_window = 20
long_window = 50

initial_balance = 10000 # USD
balance = initial_balance
position = 0 # EUR

forex_pair = 'EURUSD=X'

data = yf.download(forex_pair, period='6d', interval='1m', progress=False)

data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

for index, row in data.iterrows():
    sma_short = row['SMA_Short'].item()  # Приведение к скаляру
    sma_long = row['SMA_Long'].item()
    close_price = row['Close'].item()

    if sma_short > sma_long and position == 0:
        units_to_buy = balance // close_price
        balance -= units_to_buy * close_price
        position += units_to_buy

        print(f'{dt.datetime.now()}: Bought {units_to_buy} EUR for {close_price:.2f} USD per unit')

    elif sma_short < sma_long and position > 1:
        balance += position * close_price
        print(f'{dt.datetime.now()}: Sold {position} EUR for {close_price:.2f} USD per unit')
        position = 0

    else:
        print(f'{dt.datetime.now()}: Holding {position} EUR at {close_price:.2f} USD per unit')

final_balance = balance + position * data.iloc[-1]['Close'].item()
print(f'Final Balance: ${final_balance}')

if final_balance > initial_balance:
    print(f'Balance was increased by {(final_balance / initial_balance - 1) * 100:.2f}%')
elif final_balance < initial_balance:
    print(f'Balance was decreased by {(1 - final_balance / initial_balance) * 100:.2f}%')
else:
    print('Balance did not change!')