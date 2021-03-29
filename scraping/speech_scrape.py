from bs4 import BeautifulSoup
from urllib.request import urlopen
import json
import datetime
import pickle
import pandas as pd
from FedTools import MonetaryPolicyCommittee


def scrape_fed_statements():
    DO_SCRAPE = True
    if DO_SCRAPE:
        FOMC = MonetaryPolicyCommittee(
            historical_split=2014,
            verbose=True,
            thread_num=10)

        data = FOMC.find_statements()

        data.info()
        data.to_pickle("scraping/FOMC_statements.pkl")

    data = pd.read_pickle("scraping/FOMC_statements.pkl")

    for i in data.index:
        d = pd.to_datetime(i)
        if d.year < 2013:
            data = data.drop(i)

    data.info()

    print(data["FOMC_Statements"][0])

    data.to_pickle("scraping/FOMC_statements.pkl")


def scrape_powell_speeches():
    data_list = []
    # format_dates()

    with open("scraping/formatted_dates.txt", "r") as f:
        for line in f:
            date = line[0:line.find('\n')]
            url = F"https://www.federalreserve.gov/newsevents/speech/powell{date}.htm"

            # sends request to url
            print(url)
            page = urlopen(url)
            # reads html file as a string
            html = page.read().decode("utf-8")
            # assign string to a beautifulsoup object
            soup = BeautifulSoup(html, "html.parser")

            soup.a.decompose()

            article = soup.find('div', class_="col-xs-12 col-sm-8 col-md-8")

            paragraphs = article.find_all("p")

            text = ""

            for p in paragraphs:
                if p.getText() is not None:
                    if (p.getText().find("References") >= 0
                            or (p.find_previous("hr") is not None
                                and p.find_previous("hr")['width'] == '33%')):
                        break
                    if (p.getText().find("Watch live") >= 0
                            or p.getText().find("View speech charts and figures") >= 0
                            or p.getText().find("Accessible Version") >= 0):
                        continue

                    text = text + p.getText()

            lists = article.find_all("li")
            for l in lists:
                if l.getText() is not None:
                    if (l.getText() == "Watch live"
                            or l.getText().find("View speech charts and figures") >= 0
                            or l.getText().find("Accessible Version") >= 0):
                        continue

                    text = text + l.getText()

            print(text)

            article_dict = {
                "speaker": "powell",
                "date": F"{line[0:8]}",
                "url": url,
                "title": soup.find("h3", class_="title").string,
                "content": text
            }

            data_list.append(article_dict)

    with open("scraping/powell_data.json", "w") as outfile:
        json.dump(data_list, outfile)

    return 0


def format_dates():
    with open("scraping/speech_dates.txt", "r") as original:
        with open("scraping/formatted_dates.txt", "w") as formatted:
            for line in original:
                m = int(line[0:line.find("/")])
                d = int(line[line.find("/") + 1:line.rfind("/")])
                Y = int(line[line.rfind("/") + 1:len(line)])

                date_str = F"0{d}" if d < 10 else F"{d}"
                date_str = F"0{m}{date_str}" if m < 10 else F"{m}{date_str}"
                date_str = F"{Y}{date_str}"

                formatted.write(F"{date_str}a\n")


if __name__ == '__main__':
    # scrape_powell_speeches()
    scrape_fed_statements()
