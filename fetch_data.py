import requests
import pandas as pd
import os
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")

if not fred_api_key:
    raise EnvironmentError("FRED_API_KEY not found in environment. Did you forget to load your .env file?")

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

def get_fed_data():
    fred = Fred(api_key=fred_api_key)

    series_config = {
        'FF': ('W', 'ffill'),                 # Weekly Fed Funds Rate
        'FEDFUNDS': ('M', 'ffill'),           # Monthly Fed Funds (alt)
        'EFFR': ('D', 'mean'),                # Daily Effective Fed Funds Rate
        'CPIAUCNS': ('M', 'ffill'),           # Monthly CPI
        'USREC': ('M', 'ffill'),              # Monthly Recession Indicator
        'USEPUINDXM': ('M', 'ffill')          # Monthly EPU Index
    }

    all_series = {}

    for series_id, (original_freq, agg_method) in series_config.items():
        print(f"Fetching {series_id}...")
        data = fred.get_series(series_id)
        data.index = pd.to_datetime(data.index)

        if agg_method == 'mean':
            weekly = data.resample('W-MON').mean()
        elif agg_method == 'ffill':
            weekly = data.resample('W-MON').ffill()
        else:
            raise ValueError(f"Unknown aggregation: {agg_method}")
        
        all_series[series_id] = weekly
    
    macro_df = pd.DataFrame(all_series)
    macro_df.dropna(inplace=True)

    return macro_df


if __name__ == '__main__':
    output_dir = "data"
    kraken_df = get_kraken_ohlc()
    if kraken_df is not None:
        os.makedirs(output_dir, exist_ok=True)
        kraken_df.to_csv(os.path.join(output_dir, "weekly_btc_ohlc.csv"), index=False)
        print("Save to data/weekly_btc_ohlc.csv")
    
    '''
    Be sure to set a FRED API key in your environment before running:

    export FRED_API_KEY=your_key_here

    # Or on Windows:
    set FRED_API_KEY=your_key_here
    '''

    fed_df = get_fed_data()
    if fed_df is not None:
        os.makedirs(output_dir, exist_ok=True)
        fed_df.to_csv(os.path.join(output_dir, 'fed_macro.csv'))
        print('Saved to data/fed_macro.csv')
    
