from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import argparse
import gnp
from newspaper import Article


def get_article(url):
	article = Article(url.decode())
	article.download()
	if not article.is_downloaded:
		return -1
	article.parse()
	return article

def print_article(article):
	if article!=-1:
		print("==========="+article.title+"==========")
		print(article.text)
		print("======================================")


def decode_g_result(g):
	print("Timestamp: "+g["meta"]["timestamp"])
	if len(g["stories"])<3:
		print("Not enough stories found.")
		sys.exit()
	s0 = get_article(g["stories"][0]["link"])
	s1 = get_article(g["stories"][1]["link"])
	s2 = get_article(g["stories"][2]["link"])
	print_article(s0)
	print_article(s1)
	print_article(s2)
	return s0, s1, s2

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--query', type=str, default=None, 
						help='news query')
	args = parser.parse_args()
	if args.query == None:
		print('Please input a query using --query "Something to search"')
		sys.exit()
	g_result = gnp.get_google_news_query(args.query)
	decode_g_result(g_result)
