from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
import gnp
import pickle
from sumbasic import sum_basic
from newspaper import Article
from reader import news_reader
from sumLSTM import Trainer
from datetime import datetime

with open("../data/wikinews.json","r") as f:
	data = json.load(f)

crawled = []
titles = []
for i in range(len(data)):
	s0 = data[i]["articles"][0]["body"]
	s1 = data[i]["articles"][1]["body"]
	s2 = data[i]["articles"][2]["body"]
	crawled.append([s0]+[s1]+[s2])
	titles.append(data[i]["title"])

summary_basic = []
for i in range(len(crawled)):
	summary_basic.append(sum_basic(crawled[i], 300))

urls = []
for title in titles:
	url = "https://en.wikinews.org/wiki/"+title
	#article = Article(url).download().parse().text
	#summary_basic.append(article)
	urls.append(url)

summary_lstm = []

with open("../summary/summary.txt","r") as f:
	for i in range(len(titles)):
		f.readline()
		f.readline()
		summary_lstm.append(f.readline())
		f.readline()


def generate_html(batch_size, urls, stories, summary_basic, summary_LSTM):
	html = '<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8" /></head>\n'
	html += '<p>'+str(datetime.utcnow()) + " UTC Time \n"+'</p>'
	for i in range(batch_size):
		html += '<h1><a href="'+urls[i]+'">'+stories[i]+'</a></h1>\n'
		html += '<h2>SumBasic Method</h2>\n'
		html += '<p>'+summary_basic[i]+'</p>\n'
		html += '<h2>LSTM-based Method</h2>\n'
		html += '<p>'+summary_LSTM[i]+'</p>\n'
	with open("./index2.html", "w") as file:
		print(html, file=file)



generate_html(len(titles), urls, titles, summary_basic, summary_lstm)