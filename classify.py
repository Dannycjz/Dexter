from datetime import datetime, timedelta
from numpy.core.numeric import True_
import pandas as pd
import csv
import sklearn
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk import sent_tokenize
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
from StopWords_Generic import stopwords

stopwords = stopwords
arr = list(range(1,70))

classifiers = {
        # "BernoulliNB": BernoulliNB(alpha = 0.01),
        "ComplementNB": ComplementNB(alpha = 0.01, norm = True),
        # "MultinomialNB": MultinomialNB(alpha = 0.03),
        # "KNeighborsClassifier": KNeighborsClassifier(algorithm = 'auto', leaf_size = 1, metric = 'euclidean', n_neighbors = 50, p = 1, weights = 'uniform'),
        # "DecisionTreeClassifier": DecisionTreeClassifier(),
        # "RandomForestClassifier": RandomForestClassifier(),
        # "LogisticRegression": LogisticRegression(max_iter=1000),
        # "SGDClassifier": SGDClassifier(),
        # "AdaBoostClassifier": AdaBoostClassifier(),
        # "MLPClassifier": MLPClassifier(max_iter=1000),
        # "SVC": SVC(),
        # "NuSVC": NuSVC(),
        # "LinearSVC": LinearSVC(),
        }

tuple_classifiers = [("BernoulliNB", BernoulliNB(alpha = 0.01)),
        ("ComplementNB", ComplementNB(alpha = 0.01, norm = True)),
        ("MultinomialNB", MultinomialNB(alpha = 0.03)),
        ("KNeighborsClassifier", KNeighborsClassifier(algorithm = 'auto', leaf_size = 1, metric = 'euclidean', n_neighbors = 50, p = 1, weights = 'uniform')),
        ("DecisionTreeClassifier", DecisionTreeClassifier()),
        ("RandomForestClassifier", RandomForestClassifier()),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("SGDClassifier", SGDClassifier(loss='log')),
        ("AdaBoostClassifier", AdaBoostClassifier()),
        ("MLPClassifier", MLPClassifier(max_iter=1000)),
        ("SVC", SVC(probability=True)),
        ("NuSVC", NuSVC(probability=True)),
        #("LinearSVC", LinearSVC())
        ]

STOCK_SYMBOL = 'NDAQ'
# TAGS = ["vg", "g", "n", "b", 'vb']  # v = very, g = good, b = bad, n = neutral
TAGS = ['g', 'b']


def classify(stock_symbol):
    stock = Quote(stock_symbol, '4. close')
    data = read_data(stock)

    print("\nTable info")
    data.info()  # prints table structure to terminal

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words=stopwords, ngram_range=(1, 1))

    print("\nGenerating bag of words:")
    text_counts = cv.fit_transform(data['content'])

    text_counts = integrate_db("dataset/master_dict_filtered.csv", data, text_counts, cv)

    # tfidf_counts = TfidfVectorizer().fit_transform(data['content'])
    print(F"Matrix size: {text_counts.shape}")

    RANDOM_STATE = 123
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

    print("\nTraining Classifier:")
    # trains and predicts for all classifiers

    # Voting classifier that combines existing classifying algorithms
    # voting_classifier = VotingClassifier(estimators = tuple_classifiers, voting='soft').fit(X_train, y_train)

    # y_pred = voting_classifier.predict(X_test)
    # print("voted classifier accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
    
    highest_score = [0, ""]
    for name, sklearn_clf in classifiers.items():
        start = time.time()
        y_pred = clf.predict(X_test)
        end = time.time()
 
        print(f"{name} ({(end - start) / 60:.3} min)")
        accuracy = metrics.accuracy_score(y_test, y_pred)

    # keep track of highest accuracy
        if accuracy > highest_score[0]:
            highest_score[0] = accuracy
            highest_score[1] = name
     
        print(F"{accuracy:.2%} - {stock_symbol}")
        print(classification_report(y_test, y_pred, target_names=TAGS))

    log_result(highest_score[1], highest_score[0], stock_symbol)

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
                        lil_tc[doc_i, word_i] *= int(float(row['Pos Freq']))
                elif data['tag'][doc_i] == 'b':
                    if row['Negative'] != 'empty' and row['Negative'] in feature_dict:
                        word_i = feature_dict[row['Negative']]
                        lil_tc[doc_i, word_i] *= int(float(row['Neg Freq']))
                # if row['Word'] in feature_dict:
                #     word_i = feature_dict[row['Word']]
                #     lil_tc[doc_i, word_i] *= int(float(row['Word Freq']))

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

# Searches through classifier parameters to find the best settings
def search_parameters(sklearn_clf, stock_symbol):

    stock = Quote(stock_symbol, '4. close')
    data = read_data(stock)

    print("\nTable info")
    data.info()  # prints table structure to terminal

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words=stopwords, ngram_range=(1, 1))

    print("\nGenerating bag of words:")
    text_counts = cv.fit_transform(data['content'])

    text_counts = integrate_db("dataset/master_dict_filtered.csv", data, text_counts, cv)

    print(F"Matrix size: {text_counts.shape}")

    RANDOM_STATE = 123
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

    # Building an integrated pipeline 
    text_clf = Pipeline([
            # ('vect', CountVectorizer(tokenizer=tokens.tokenize, stop_words=stopwords)),
            ('tfidf', TfidfTransformer()),
            ('clf', sklearn_clf),
        ])

    # Defines parameters for gridsearch 
    parameters = {
            # 'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3, 1e-4), 
            'clf__norm': (True, False),
            # 'clf__criterion': ('gini', 'entropy'),
            # 'clf__splitter': ('best', 'random'), 
            # 'clf__max_depth': (None, 1, 5, 10, 20),
            # 'clf__min_samples_split': (1, 5, 10, 20),
            # 'clf__min_samples_leaf': (1, 5, 10, 20),
            # 'clf__min_weight_fraction_leaf':(0.0, 1e-2, 1e-3, 1e-4),
            # 'clf__max_features': (None, 'auto', 'sqrt', 'log2', 1, 5, 10, 20), 
            # 'clf__random_state': (None, 1, 5, 10, 20), 
            # 'clf__max_leaf_nodes': (None, 1, 5, 10, 20), 
            # 'clf__min_impurity_decrease': (0.0, 1e-2, 1e-3, 1e-4),
            # 'clf__ccp_alpha': (0.0, 1e-2, 1e-3, 1e-4)
            }

    # Training Gridsearch classifier
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

    clf = gs_clf.fit(X_train, y_train)

    clf.best_score_

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, clf.best_params_[param_name]))


if __name__ == '__main__':

    # classify("JPM")

    tickers = ["NFLX"]
    for t in tickers:
        classify(t)
        search_parameters(ComplementNB(), t)

