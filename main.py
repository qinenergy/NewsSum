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
		s0 = -1
	try:
		s1 = get_article(g["stories"][1]["link"]).text
	except:
		s1 = -1
	return [s0]+[s1]

def generate_html(batch_size, urls, stories, summary_basic, summary_LSTM):
	html = '<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8" /></head>\n'
	html += '<p>'+str(datetime.utcnow()) + " UTC Time \n"+'</p>'
	for i in range(batch_size):
		html += '<h1><a href="'+urls[i]+'">'+stories[i]+'</a></h1>\n'
		html += '<h2>SumBasic Method</h2>\n'
		html += '<p>'+summary_basic[i]+'</p>\n'
		html += '<h2>LSTM-based Method</h2>\n'
		html += '<p>'+summary_LSTM[i]+'</p>\n'
	with open("./docs/index.html", "w") as file:
		print(html, file=file)

if __name__ == "__main__":
	# Parameters (should be same as trained model's)
	max_doc_length = 60
	max_sen_length = 50
	batch_size = 20
	SumBasicMaxLen = 300
	# Crawl News
	b = gnp.get_google_news(gnp.EDITION_ENGLISH_US, geo='New York,USA')
	print("News topics loaded")
	stories_ = [b["stories"][i]["title"].decode().replace("...", "") for i in range(len(b["stories"]))]
	urls_ = [b["stories"][i]["link"].decode() for i in range(len(b["stories"]))]
	crawled = []
	urls = []
	stories = []
	for i in range(len(b["stories"])):
		article = decode_g_result(stories_[i])
		if (-1 not in article) and ("" not in article):
			crawled.append(article)
			urls.append(urls_[i])
			stories.append(stories_[i])
		if len(crawled)>batch_size:
			break
	print("Articles loaded")
	# Generate summary based on 
	summary_basic = []
	for i in range(batch_size):
		summary_basic.append(sum_basic(crawled[i], SumBasicMaxLen))
	print("SumBasic generated")
	# Neural Network Method
	summary_LSTM = []
	embedding_path = "./data/wordembeddings-dim100.word2vec"
	with open('./data/word_vocab.pkl', 'rb') as input:
		word_vocab = pickle.load(input)

	nr = news_reader(max_doc_length, max_sen_length, word_vocab, batch_size, crawled)
	initializer = tf.random_uniform_initializer(-0.05, 0.05)
	trainer = Trainer(initializer, word_vocab, embedding_path,
					  max_doc_length = max_doc_length,
					  max_sen_length = max_sen_length,
					  batch_size = batch_size)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		# Restore trained model
		saver.restore(sess, "./model.ckpt")
		print("LSTM Model loaded")
		# Predictions
		x, o = nr.next_batch()
		logits = sess.run([trainer.logits],{trainer.input: x})
		logits = np.array(logits[0])
		pred = np.argmax(np.transpose(logits, (1,0,2)), axis=2)
		for j in range(batch_size):
			summary = ""
			for k, label in enumerate(pred[j]):
				if label==1:
					summary += o[j][k] + " "
			summary_LSTM.append(summary)

	# Generate
	generate_html(batch_size, urls, stories, summary_basic, summary_LSTM)
	print("index.html generated")
	