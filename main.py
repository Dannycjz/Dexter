from pygooglenews import GoogleNews


def main():
    print("hello world")


    gn = GoogleNews()

    top = gn.top_news()

    for i in range(10):
        print(top['entries'][i])

    return 0


if __name__ == '__main__':
    main()