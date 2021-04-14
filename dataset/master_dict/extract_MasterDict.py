# system
import csv
from tqdm import tqdm
# lib
from StopWords_Generic import stopwords


def extract_most_common(max_features):
    positive_words = []
    negative_words = []
    words = []

    with open("LM_MasterDict.csv", "r") as m_dict:
        reader = csv.DictReader(m_dict)
        stop_words = stopwords

        for row in tqdm(reader):
            if int(row['Positive']) > 0:
                positive_words.append((int(row['Word Count']) * float(row['Word Proportion']), row['Word']))
            if int(row['Negative']) > 0:
                negative_words.append((int(row['Word Count']) * float(row['Word Proportion']), row['Word']))
            if row['Word'].lower() not in stop_words:
                words.append((int(row['Doc Count']), row['Word']))

    positive_words.sort(key=lambda x: x[0])
    positive_words.reverse()

    negative_words.sort(key=lambda x: x[0])
    negative_words.reverse()

    words.sort(key=lambda x: x[0])
    words.reverse()

    return positive_words[0:max_features], negative_words[0:max_features], words[0:max_features]


def write_to_file(path, pos_w, neg_w, w):
    with open(path, "w") as f:
        field_names = ['Index', 'Positive', "Pos Freq", 'Negative', "Neg Freq", 'Word', "Word Freq"]
        writer = csv.DictWriter(f, fieldnames=field_names)

        writer.writeheader()
        for i in range(len(w)):
            writer.writerow({
                'Index': i,
                'Positive': pos_w[i][1].lower() if i < len(pos_w) else "empty",
                'Pos Freq': pos_w[i][0] if i < len(pos_w) else 1,
                'Negative': neg_w[i][1].lower() if i < len(neg_w) else "empty",
                'Neg Freq': neg_w[i][0] if i < len(neg_w) else 1,
                'Word': w[i][1].lower(),
                'Word Freq': w[i][0]
            })


if __name__ == '__main__':
    positive_words, negative_words, words = extract_most_common(2000)
    write_to_file("master_dict_filtered.csv", positive_words, negative_words, words)










