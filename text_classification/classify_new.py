from datetime import datetime, timedelta
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
from tqdm import tqdm
from scipy import sparse
import pickle

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random
from StockAPI import Quote
import time

classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(alpha=0.3),
}


TAGS = ['g', 'b']
DO_GET_QUOTES_FROM_API = False

# TODO: classify should be auto updated when classify.py is updated.
# TODO: classify new should directly call class.py, by passing in args that do that


def classify(speech, stock_symbol):

    data = pickle.load(open(f"dataset/Fed/powell_{stock_symbol}.pkl", 'rb'))
    cv = pickle.load(open("text_classification/pickles/cv.sav", 'rb'))

    print("\nGenerating bag of words:")
    text_counts = cv.transform([speech])

    data.info()  # prints table structure to terminal

    text_counts = integrate_db("dataset/master_dict/master_dict_filtered.csv", data, text_counts)

    CLF_NAME = 'ComplementNB'
    PERCENT = "68.92%"

    clf = pickle.load(open(f"text_classification/pickles/{CLF_NAME}_{PERCENT}_{stock_symbol}.sav", 'rb'))

    return clf.predict_proba(text_counts)


def integrate_db(db_path, data, text_counts):

    length = text_counts.shape[0]
    feature_dict = pickle.load(open("text_classification/pickles/feature_dict.sav", 'rb'))

    lil_tc = sparse.lil_matrix(text_counts)  # converts text counts from csr matric to lil matrix to increase efficiency

    with open(db_path, 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=length)  # makes a new progress bar

        # iterate through all documents
        for doc_i in range(length):
            # iterate through each word in filtered_master_dict
            for row in reader:
                # for positive words check if they exist in text_counts' features
                if data['tag'][doc_i] == 'g':
                    if row['Positive'] != 'empty' and row['Positive'] in feature_dict:
                        word_i = feature_dict[row['Positive']]
                        # multiplies entry by frequency in master_dict filtered if document is tagged as 'g'
                        lil_tc[doc_i, word_i] *= float(row['Pos Freq']) * 10
                elif data['tag'][doc_i] == 'b':
                    if row['Negative'] != 'empty' and row['Negative'] in feature_dict:
                        word_i = feature_dict[row['Negative']]
                        lil_tc[doc_i, word_i] *= float(row['Neg Freq']) * 10

            pbar.update(1)
        pbar.close()

    return sparse.csr_matrix(lil_tc) # converts lil matrix back to csr



if __name__ == '__main__':

    tickers = ["PLUG"]
    for t in tickers:
        classify(t)
