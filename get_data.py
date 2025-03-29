import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import requests
import ta
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
import time
def get_data(symbol, interval):
    # Fetch historical BTC/USD data from Binance API
    
    url = 'https://api.binance.com/api/v3/klines'
    limit = 1000  # Max per request
    end_time = int(datetime.now().timestamp() * 1000)  # Current time in ms
    start_time = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)  # 2 years ago

    all_data = []

    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data += data

        # If we received less than the limit, we're done
        if len(data) < limit:
            break

        # Move the start time forward to avoid overlap
        start_time = data[-1][0] + 1
        time.sleep(0.2)  # Respect rate limits

    df = pd.DataFrame(all_data,
                      columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                               'Quote_asset_volume', 'Number_of_trades',
                               'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])

    df['Date'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Date', inplace=True)

    return df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})



def calculate_indicators(data):
    # Calculate SMA, EMA, and Bollinger Bands
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA100'] = data['Close'].rolling(window=100).mean()
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['Upper_BB'], data['Lower_BB'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(
        window=20).std(), data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    # Calculate MACD and Signal Line
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    # Calculate RSI
    delta = data['Close'].diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    # Calculate support and resistance levels for 1D timeframe
    data['High_Rolling'] = data['High'].rolling(50).max()
    data['Low_Rolling'] = data['Low'].rolling(50).min()
    data['Close_Shift'] = data['Close'].shift(1)
    data['Pivot'] = (data['High_Rolling'] + data['Low_Rolling'] + data['Close_Shift']) / 3
    data['R1'] = 2 * data['Pivot'] - data['Low_Rolling']
    data['S1'] = 2 * data['Pivot'] - data['High_Rolling']
    data['R2'] = data['Pivot'] + (data['High_Rolling'] - data['Low_Rolling'])
    data['S2'] = data['Pivot'] - (data['High_Rolling'] - data['Low_Rolling'])
    # Calculate MFI
    data['MFI'] = ta.volume.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], window=14)

    
btc_data = get_data(symbol = 'BTCUSDT',interval = '4h' )
eth_data = get_data(symbol = 'ETHUSDT',interval = '4h' )
solana_data = get_data(symbol = 'SOLUSDT',interval = '4h' )

btc_data.to_csv("btc_data.csv", index=False)
eth_data.to_csv("eth_data.csv", index=False)
solana_data.to_csv("solana_data.csv", index=False)