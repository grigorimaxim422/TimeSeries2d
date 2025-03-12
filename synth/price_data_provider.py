import requests
import pytz
from synth.utils.helpers import from_iso_to_unix_time
from datetime import datetime, timezone
import pandas as pd

class PriceDataProvider:
    BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

    TOKEN_MAP = {"BTC": "Crypto.BTC/USD", "ETH": "Crypto.ETH/USD"}

    one_day_seconds = 24 * 60 * 60
    two_day_seconds = 48 * 60 * 60
    one_week_seconds = one_day_seconds * 7

    def __init__(self, token):
        self.token = self._get_token_mapping(token)    
    
    def fetch_afterward(self, start_time, step=300):
        start_time = start_time
        end_time = start_time + self.one_day_seconds
        params = {
            "symbol": self.token,
            "resolution": step//60,
            "from": start_time,
            "to": end_time,
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()        

        data = response.json()
        price_data = data['c']
        price_data = price_data[1:]
        df = pd.DataFrame(data)
        df = df.drop(columns=['s', 'h','l','o','v'])
        df = df.iloc[1:]
        df.reset_index(drop=True, inplace=True)
        df = df.rename(columns={
            't': 'timestamp',                        
            'c': 'price',            
        })        
        
        df ['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return price_data, df 
    
        
    def fetch_after(self, start_time:str, step=300):
        start_time = from_iso_to_unix_time(start_time)
        end_time = start_time + self.one_day_seconds
        params = {
            "symbol": self.token,
            "resolution": step//60,
            "from": start_time,
            "to": end_time,
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()        

        data = response.json()
        transformed_data = data['c']
        transformed_data = transformed_data[1:]
        return transformed_data                    
    
    def fetch_last_day(self, time_point:str, step=300):
        end_time = from_iso_to_unix_time(time_point)
        start_time = end_time - self.one_day_seconds
        params = {
            "symbol": self.token,
            "resolution": step//60,
            "from": start_time,
            "to": end_time,
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()        
        transformed_data = self._transform_to_csv(data, start_time, step)
        return transformed_data
    
    def fetch_csv(self, time_point:str, step=300):
        end_time = from_iso_to_unix_time(time_point)
        start_time = end_time - self.two_day_seconds
        params = {
            "symbol": self.token,
            "resolution": step//60,
            "from": start_time,
            "to": end_time,
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()        
        transformed_data = self._transform_to_csv(data, start_time, step)
        return transformed_data, end_time

    def fetch_data(self, time_point: str):
        """
        Fetch real prices data from an external REST service.
        Returns an array of time points with prices.

        :return: List of dictionaries with 'time' and 'price' keys.
        """

        end_time = from_iso_to_unix_time(time_point)
        start_time = end_time - self.one_day_seconds

        params = {
            "symbol": self.token,
            "resolution": 1,
            "from": start_time,
            "to": end_time,
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()        
        transformed_data = self._transform_data(data, start_time)

        return transformed_data

    @staticmethod
    def _transform_to_csv(data, start_time,step):
        df = pd.DataFrame(data)
        df = df.iloc[1:].reset_index(drop=True)
        df = df.rename(columns={
            't': 'timestamp',                        
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            'o': 'Open',
            'v': 'volume'
        })
        df = df.drop(columns=["s"])       
        df.to_csv('tmp.csv', index=False)
        newdata = pd.read_csv('tmp.csv')
        # newdata['timestamp'] = pd.to_datetime(newdata['timestamp'])
        # newdata['timestamp'] = newdata['timestamp'].dt.strftime()
        newdata['timestamp'] = pd.to_datetime(newdata['timestamp'], unit='s')
        # newdata = newdata.sort_index()
        # newdata = newdata.sort_values(by="timestamp")
        newdata.set_index('timestamp', inplace=True)
        return newdata
    
    @staticmethod
    def _transform_to_line(data, start_time,step):
        if data is None or len(data) == 0:
            return []

        timestamps = data["t"]
        close_prices = data["c"]

        transformed_data = []

        for t, c in zip(timestamps, close_prices):
            if (
                t >= start_time and (t - start_time) % step == 0
            ):  # 300s = 5 minutes
                transformed_data.append(c)

        return transformed_data
        
        
    @staticmethod
    def _transform_data(data, start_time):
        if data is None or len(data) == 0:
            return []

        timestamps = data["t"]
        close_prices = data["c"]

        transformed_data = []

        for t, c in zip(timestamps, close_prices):
            if (
                t >= start_time and (t - start_time) % 300 == 0
            ):  # 300s = 5 minutes
                transformed_data.append(
                    {
                        "time": datetime.fromtimestamp(
                            t, timezone.utc
                        ).isoformat(),
                        "price": float(c),
                    }
                )

        return transformed_data

    @staticmethod
    def _get_token_mapping(token: str) -> str:
        """
        Retrieve the mapped value for a given token.
        If the token is not in the map, raise an exception or return None.
        """
        if token in PriceDataProvider.TOKEN_MAP:
            return PriceDataProvider.TOKEN_MAP[token]
        else:
            raise ValueError(f"Token '{token}' is not supported.")
