import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from nltk import sent_tokenize
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import random


STOCK_SYMBOL = "SPY"
TAGS = ["vg", "g", "n", "b", 'vb']  # v = very, g = good, b = bad, n = neutral


def classify():
    data = read_data()

    print("\nTable info")
    data.info()  # prints table structure to terminal

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 2))

    print("\nGenerating bag of words:")
    text_counts = cv.fit_transform(tqdm(data['content']))
    tfidf_counts = TfidfTransformer().fit_transform(text_counts)
    print(F"Matrix size: {text_counts.shape}")

    RANDOM_STATE = 123
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['tag'], test_size=0.3, random_state=RANDOM_STATE)

    clf = AdaBoostClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nResults:")
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(F"{accuracy:.2%} - {STOCK_SYMBOL}")

    print(classification_report(y_test, y_pred, target_names=TAGS))


def read_data():
    print("reading data...")
    data = pd.read_json("scraping/powell_data.json")

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

    data.to_pickle("main_dataset.pkl")

    print("\n5 docs:")
    for i in [0, 1000, 2000, 3000, 4000, 5000]:
        print(F"{data['date'][i]}: {data['tag'][i]}  \t|  {data['title'][i]}")

    return data


if __name__ == '__main__':
    classify()
