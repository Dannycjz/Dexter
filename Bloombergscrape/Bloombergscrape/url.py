"""
The archive is nicely structured, thus the purpose of this script is 
to generate a txt file containing all the urls of the blog-posts 
published between 1 January 2008 and 15 August 2014.
In order to achieve this goal I implemented the following steps:
1- generate the urls of all the months in the time interval
2- generate the urls of all the days for each month
3- scrape each of the day-urls and get all the urls of the 
   posts published on that specific day.
4- repeat for all the days on which something was published 
"""
import scrapy
import urllib.request
 
def businessWeekUrl():
    totalWeeks = []
    totalPosts = []
    url = 'http://www.businessweek.com/archive/news.html#r=404'
    data = urllib.request.urlopen(url).read()
    hxs = scrapy.Selector(text=data)
    
    months = hxs.xpath('//ul/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news.html')    
    admittMonths = 12*(2013-2007) + 8
    months = months[:admittMonths]
 
    for month in months:
        data = urllib.request.urlopen(month).read()
        hxs = scrapy.Selector(text=data)
        weeks = hxs.xpath('//ul[@class="weeks"]/li/a').re('http://www.businessweek.com/archive/\\d+-\\d+/news/day\\d+\.html')
        totalWeeks += weeks
    
    for week in totalWeeks:
        data = urllib.request.urlopen(week).read()
        hxs = scrapy.Selector(text=data)
        posts = hxs.xpath('//ul[@class="archive"]/li/h1/a/@href').extract()
        totalPosts += posts
    
    with open("businessweek.txt", "a") as myfile:
        for post in totalPosts:
            post = post + '\n'
            myfile.write(post)
 
businessWeekUrl()