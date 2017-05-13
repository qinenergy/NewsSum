from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import gnp
from sumbasic import sum_basic
from newspaper import Article


def get_article(url):
	article = Article(url.decode())
	article.download()
	if not article.is_downloaded:
		return -1
	article.parse()
	return article

def decode_g_result(title):
	g = gnp.get_google_news_query(title)
	try:
		s0 = get_article(g["stories"][0]["link"]).text
	except:
		s0 = "Not enough article found under title" + title
	try:
		s1 = get_article(g["stories"][1]["link"]).text
	except:
		s0 = "Not enough article found under title" + title
	return [s0]+[s1]

if __name__ == "__main__":
	num = 20
	b = gnp.get_google_news(gnp.EDITION_ENGLISH_US)
	stories = [b["stories"][i]["title"].decode().replace("...", "") for i in range(num)]
	urls = [b["stories"][i]["link"].decode() for i in range(num)]
	crawled = []
	summary_basic = []
	for i in range(num):
		crawled.append(decode_g_result(stories[i]))
	for i in range(num):
		summary_basic.append(sum_basic(crawled[i], 150))
	
	html = ""
	for i in range(num):
		html += '<h1><a href="'+urls[i]+'">'+stories[i]+'</a></h1>\n'
		html += '<h2>SumBasic Method</h2>\n'
		html += '<p>'+summary_basic[i]+'</p>\n'
		html += '<h2>LSTM-based Method</h2>\n'

	with open("./docs/index.html", "w") as file:
		print(html, file=file)