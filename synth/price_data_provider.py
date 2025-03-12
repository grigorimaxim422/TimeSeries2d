import requests

from synth.utils.helpers import from_iso_to_unix_time
from datetime import datetime, timezone
import pandas as pd

class PriceDataProvider:
    BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

    TOKEN_MAP = {"BTC": "Crypto.BTC/USD", "ETH": "Crypto.ETH/USD"}

    one_day_seconds = 24 * 60 * 60
    one_week_seconds = one_day_seconds * 7

    def __init__(self, token):
        self.token = self._get_token_mapping(token)
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
        transformed_data = []
        for c in zip(data["c"]):
            transformed_data.append(c)
        
        return transformed_data                    
        
    def fetch_csv(self, time_point:str, step=300):
        end_time = from_iso_to_unix_time(time_point)
        start_time = end_time - self.one_week_seconds
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
        df = df.rename(columns={
            't': 'time',                        
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            'o': 'Open',
            'v': 'volume'
        })
        df = df.drop(columns=["s"])
        # Select only the relevant columns        
        df.to_csv('tmp.csv', index=False)
        newdata = pd.read_csv('tmp.csv')
        # newdata["time"] = pd.to_datetime(newdata["time"])
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
