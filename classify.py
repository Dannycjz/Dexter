import datetime
import pickle
import pandas as pd
from FedTools import MonetaryPolicyCommittee

STOCK_SYMBOL = "SPY"


def classify():
    read_data()


def read_data():
    print("reading data...")
    data = pd.read_json("scraping/powell_data.json")
    data.info()  # prints table structure to terminal
    tags = []
    for i in data.index:
        date_str = str(data["date"][i])
        Y = date_str[0:4]
        m = date_str[4:6]
        d = date_str[6:8]

        # TODO: use your function. The date of the speech is extracted above.
        # TODO: store the difference of stock prices in delta for 1 day before and 1 day after
        # TODO: make a check for the dates so to make sure the 2 days are trading days
        # TODO: possible soln: keep incrementing days forward/backward until valid

        stock = STOCK_SYMBOL

        delta = 10  # TODO: change this
        tags.append(delta)

    data.insert(5, "tag", tags, True)
    data.to_pickle("main_dataset.pkl")

    for i in range(5):
        print(F"{data['date'][i]}: {data['tag'][i]}\t|\t{data['title'][i]}")


if __name__ == '__main__':
    classify()
