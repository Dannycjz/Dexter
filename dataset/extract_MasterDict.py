# system
import pickle
import csv
from tqdm import tqdm
# lib
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import ngrams
import re


def extract_most_common(max_features):
    positive_words = []
    negative_words = []

    with open("LM_MasterDict.csv", "r") as m_dict:
        reader = csv.DictReader(m_dict)

        for row in tqdm(reader):
            if int(row['Positive']) > 0:
                positive_words.append((int(row['Word Count']) * float(row['Word Proportion']), row['Word']))
            if int(row['Negative']) > 0:
                negative_words.append((int(row['Word Count']) * float(row['Word Proportion']), row['Word']))

    positive_words.sort(key=lambda x: x[0])
    positive_words.reverse()

    negative_words.sort(key=lambda x: x[0])
    negative_words.reverse()

    return positive_words[0:max_features], negative_words[0:max_features]


def write_to_file(path, pos_w, neg_w):
    with open(path, "w") as f:
        field_names = ['Index', 'Positive', "Pos Freq", 'Negative', "Neg Freq"]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        for i in range(len(pos_w)):
            writer.writerow({
                'Index': i,
                'Positive': pos_w[i][1].lower(),
                'Pos Freq': pos_w[i][0],
                'Negative': neg_w[i][1].lower(),
                'Neg Freq': neg_w[i][0]
            })


if __name__ == '__main__':
    positive_words, negative_words = extract_most_common(2000)
    write_to_file("master_dict_filtered.csv", positive_words, negative_words)










