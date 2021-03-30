from alpha_vantage.timeseries import TimeSeries

# Input API Key
key = 'RG7MF5NCP690TYAJ'

# Define stock price type
tp = '4. close'

# Looks up stock info based on given time
def lookup(symbol, date1, date2):
    ts = TimeSeries(key)
    stock, meta = ts.get_daily(f"{symbol}")
    date1_prices = stock[f'{date1}']
    date2_prices = stock[f'{date2}']
    stock_price_before = date1_prices[f'{tp}']
    stock_price_after = date2_prices[f'{tp}']

    # Compares stock prices before and after
    if stock_price_after > stock_price_before: 
        print('True')
        return True 
    else:
        print('false')
        return False
























































    
    
    

