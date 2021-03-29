import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import random


STOCK_SYMBOL = "SPY"
TAGS = ["vg", "g", "n", "b", 'vb']  # v = very, g = good, b = bad, n = neutral


def classify():
    data = read_data()

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 1))
    text_counts = cv.fit_transform(tqdm(data['content']))
    tfidf_counts = TfidfTransformer().fit_transform(text_counts)

    RANDOM_STATE = 999
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tags'], test_size=0.3, random_state=RANDOM_STATE)

    clf = AdaBoostClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(F"{accuracy:.2%} - {STOCK_SYMBOL}")

    print(classification_report(y_test, y_pred, target_names=TAGS))


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
        temp = [2, -2]
        r_num = random.random() * random.choice(temp)

        delta = r_num  # TODO: change this. if possible store this as % change

        BIG_VAL = 1
        SMALL_VAL = 0.1

        if delta > BIG_VAL:
            tag = TAGS[0]
        elif delta > SMALL_VAL:
            tag = TAGS[1]
        elif delta < -1 * BIG_VAL:
            tag = TAGS[4]
        elif delta < -1 * SMALL_VAL:
            tag = TAGS[3]
        else:
            tag = TAGS[2]

        tags.append(tag)

    data.insert(5, "tags", tags, True)
    data.to_pickle("main_dataset.pkl")

    for i in range(5):
        print(F"{data['date'][i]}: {data['tags'][i]}\t|\t{data['title'][i]}")

    return data


if __name__ == '__main__':
    classify()
