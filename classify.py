from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from nltk import sent_tokenize
from tqdm import tqdm

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random
from StockAPI import Quote
import time

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=10000),
    "SGDClassifier":SGDClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
}

STOCK_SYMBOL: str = "SPY"
TAGS = ["vg", "g", "n", "b", 'vb']  # v = very, g = good, b = bad, n = neutral


def classify():
    stock = Quote(STOCK_SYMBOL, '4. close')
    data = read_data(stock)

    print("\nTable info")
    data.info()  # prints table structure to terminal

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 2))

    print("\nGenerating bag of words:")
    text_counts = cv.fit_transform(tqdm(data['content']))
    # tfidf_counts = TfidfTransformer().fit_transform(text_counts)
    print(F"Matrix size: {text_counts.shape}")

    RANDOM_STATE = 123
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

    print("\nTraining Classifier:")
    # trains and predicts for all classifiers
    for name, sklearn_clf in classifiers.items():
        start = time.time()
        clf = sklearn_clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()

        print(f"\nResults: - {name}")
        print(F"elapsed time: {(end - start) / 60:.3} min")
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(F"{accuracy:.2%} - {STOCK_SYMBOL}")
        print(classification_report(y_test, y_pred, target_names=TAGS))


def read_data(stock: Quote):
    print("reading speeches...")
    data = pd.read_json("dataset/powell_data.json")

    tags = []
    print("reading quotes...")
    for i in tqdm(data.index):
        date_str = str(data["date"][i])
        Y = date_str[0:4]
        m = date_str[4:6]
        d = date_str[6:8]

        speech_date = datetime.strptime(F"{Y}-{m}-{d}", '%Y-%m-%d')
        date1 = speech_date - timedelta(days=1)
        date1 = datetime.strftime(date1, '%Y-%m-%d')
        date2 = datetime.strftime(speech_date, '%Y-%m-%d')

        delta = stock.lookup(date1, date2)

        BIG_VAL = 1
        SMALL_VAL = 0.3

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

    data.insert(5, "tag", tags, True)

    sentence_list = []
    for i in range(len(data)):
        sentences = sent_tokenize(data['content'][i])
        for s in sentences:
            temp_dict = {
                'date': data['date'][i],
                'title': data['title'][i],
                'content': s,
                'tag': data['tag'][i]
            }
            sentence_list.append(temp_dict)

    data = pd.DataFrame(sentence_list)

    data.to_pickle("dataset/main_dataset.pkl")

    print("\n5 docs:")
    for i in [0, 1000, 2000, 3000, 4000, 5000]:
        print(F"{data['date'][i]}: {data['tag'][i]}  \t|  {data['title'][i]}")

    return data


if __name__ == '__main__':
    classify()
