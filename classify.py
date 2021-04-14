from datetime import datetime, timedelta
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from nltk import sent_tokenize, SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from tqdm import tqdm
from scipy import sparse
import nltk

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier
)
from sklearn.svm import SVC, LinearSVC, NuSVC
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
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SGDClassifier": SGDClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "MLPClassifier": MLPClassifier(max_iter=1000),
        "SVC": SVC(),
        "NuSVC": NuSVC(),
        "LinearSVC": LinearSVC(),
        }

tuple_classifiers = [("BernoulliNB", BernoulliNB()),
        ("ComplementNB", ComplementNB()),
        ("MultinomialNB", MultinomialNB()),
        ("KNeighborsClassifier", KNeighborsClassifier()),
        # ("DecisionTreeClassifier", DecisionTreeClassifier()),
        ("RandomForestClassifier", RandomForestClassifier()),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        # ("SGDClassifier", SGDClassifier()),
        # ("AdaBoostClassifier", AdaBoostClassifier()),
        # ("MLPClassifier", MLPClassifier(max_iter=1000)),
        # ("SVC", SVC()),
        # ("NuSVC", NuSVC()),
        # ("LinearSVC", LinearSVC())
        ]

STOCK_SYMBOL = 'PLUG'
# TAGS = ["vg", "g", "n", "b", 'vb']  # v = very, g = good, b = bad, n = neutral
TAGS = ['g', 'b']


def classify(stock_symbol):
    stock = Quote(stock_symbol, '4. close')
    data = read_data(stock)

    print("\nTable info")
    data.info()  # prints table structure to terminal

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 1))

    print("\nGenerating bag of words:")
    text_counts = cv.fit_transform(data['content'])

    # text_counts = integrate_db("dataset/master_dict_filtered.csv", data, text_counts, cv)

    # tfidf_counts = TfidfTransformer().fit_transform(text_counts)
    print(F"Matrix size: {text_counts.shape}")

    RANDOM_STATE = 123
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

    print("\nTraining Classifier:")

    # trains and predicts for all classifiers

    # Voting classifier that combines existing classifying algorithms
    voting_classifier = VotingClassifier(estimators = tuple_classifiers, voting='hard').fit(X_train, y_train)

    y_pred = voting_classifier.predict(X_test)
    print(F"voted classifier accuracy: {metrics.accuracy_score(y_test, y_pred): .2%}")

    log_result("voting", metrics.accuracy_score(y_test, y_pred), stock_symbol)
    
    highest_score = [0, ""]
    for name, sklearn_clf in classifiers.items():
        start = time.time()
        clf = sklearn_clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time()

        print(f"{name} ({(end - start) / 60:.3} min)")
        accuracy = metrics.accuracy_score(y_test, y_pred)
        if accuracy > highest_score[0]:
            highest_score[0] = accuracy
            highest_score[1] = name

        print(F"{accuracy:.2%} - {stock_symbol}")
        print(classification_report(y_test, y_pred, target_names=TAGS))

    # log_result(highest_score[1], highest_score[0], stock_symbol)


def integrate_db(db_path, data, text_counts, cv: CountVectorizer):
    feature_list = cv.get_feature_names()
    length = text_counts.shape[0]
    # translates list of features in dict {word => index}
    feature_dict = {feature_list[i]: i for i in range(0, len(feature_list))}

    lil_tc = sparse.lil_matrix(text_counts)

    # TODO: make sure textcounts is actually being updated
    # TODO: somehow integrate with 2 gram words as well

    with open(db_path, 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=length)

        for doc_i in range(length):
            for row in reader:
                if data['tag'][doc_i] == 'g':
                    if row['Positive'] != 'empty' and row['Positive'] in feature_dict:
                        word_i = feature_dict[row['Positive']]
                        lil_tc[doc_i, word_i] *= int(row['Pos Freq'])
                elif data['tag'][doc_i] == 'b':
                    if row['Negative'] != 'empty' and row['Negative'] in feature_dict:
                        word_i = feature_dict[row['Negative']]
                        lil_tc[doc_i, word_i] *= int(row['Neg Freq'])
                if row['Word'] in feature_dict:
                    word_i = feature_dict[row['Word']]
                    lil_tc[doc_i, word_i] *= int(row['Word Freq'])

            pbar.update(1)
        pbar.close()

    return sparse.csr_matrix(lil_tc)


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

        if delta > 0:
            tag = TAGS[0]
        else:
            tag = TAGS[1]

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


def log_result(clf_name, score, stock_symbol):
    with open("results.txt", "a") as f:
        f.write(F"{stock_symbol}: {score:.2%} - {clf_name}\n")


if __name__ == '__main__':

    # classify("JPM")

    tickers = ["NFLX"]
    for t in tickers:
        classify(t)

