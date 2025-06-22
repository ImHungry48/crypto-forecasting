import requests
import pandas as pd
import os

def get_kraken_ohlc(pair='XBTUSD', interval=10080):
    url = "https://api.kraken.com/0/public/OHLC"
    params = {'pair': pair, 'interval': interval}

    response = requests.get(url, params=params).json()

    if response['error']:
        print("Error:", response['error'])
        return None

    pair_key = [key for key in response['result'].keys() if key != 'last'][0]
    data = response['result'][pair_key]

    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='s') # TODO: make sure this is necessary
    for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
        df[col] = df[col].astype(float)
    df['count'] = df['count'].astype(int)

    return df

if __name__ == '__main__':
    df = get_kraken_ohlc()
    if df is not None:
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "weekly_btc_ohlc.csv"), index=False)
        print("Save to data/weekly_btc_ohlc.csv")