from alpha_vantage.timeseries import TimeSeries
from API_KEY import API_KEY
from datetime import datetime, timedelta

# Input API Key
key = API_KEY

class Quote():
    def __init__(self, symbol, tp):
        self.symbol = symbol
        # Define stock price type
        self.tp = tp

        ts = TimeSeries(key)
        stock, meta = ts.get_daily(f"{self.symbol}", outputsize="full")
        self.stock_history = stock

    # Looks up stock info based on given time
    def lookup(self, date1, date2):
        date1_prices = 0
        date2_prices = 0

        # ensure dates entered are trading days
        date_is_bad = True
        # print(date1)
        while date_is_bad:
            try:
                date1_prices = self.stock_history[f'{date1}']
                date_is_bad = False
            except KeyError:
                datetime_object = datetime.strptime(date1, '%Y-%m-%d')
                datetime_object -= timedelta(days=1)
                if datetime_object.year < 2013:
                    raise Exception("No valid quotes before 2013")
                date1 = datetime.strftime(datetime_object, '%Y-%m-%d')

        date_is_bad = True
        while date_is_bad:
            try:
                date2_prices = self.stock_history[f'{date2}']
                date_is_bad = False
            except KeyError:
                datetime_object = datetime.strptime(date2, '%Y-%m-%d')
                datetime_object += timedelta(days=1)
                date2 = datetime.strftime(datetime_object, '%Y-%m-%d')

        stock_price_before = date1_prices[f'{self.tp}']
        stock_price_after = date2_prices[f'{self.tp}']

        # Compares stock prices before and after
        # return stock_price_after > stock_price_before:
        delta = float(stock_price_after) - float(stock_price_before)
        # print(round(delta, 3))
        return round(delta, 3)


if __name__ == '__main__':
    s = Quote("SPY", '4. close')
    s.lookup("2020-10-29", "2021-03-26")











































    
    
    

