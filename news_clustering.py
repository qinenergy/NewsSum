import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer

#load data
import json
with open("./data/news_set.json") as data_file:
    news_set = json.load(data_file)

data_articles = []
data_date = []
data_title = []
data_url = []
data_image = []
data_label = []
for label, news_group in enumerate(news_set):
    for news_article in news_group['articles']:
        data_articles.append(news_article['body'])
        data_date.append(news_article['published'])
        data_title.append(news_article['title'])
        data_url.append(news_article['url'])
        data_image.append(news_article['body'])
        data_label.append(label)
print len(data_articles)

# <-----------TF-IDF---------->
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

#dict 
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in data_articles:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
print len(set(totalvocab_tokenized))
print len(set(totalvocab_stemmed))
#75908
#56116

#vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=200000,
                                   min_df=0.05, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
%time tfidf_matrix = tfidf_vectorizer.fit_transform(data_articles) #fit the vectorizer to synopses
print(tfidf_matrix.shape)
#CPU times: user 2min 11s, sys: 2.24 s, total: 2min 13s
#Wall time: 2min 15s
#(4413, 1108)

#search number of clusters
in_cluster_dis = []
for k in xrange(1400, 1650, 10):
    print("")
    km_mini = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, init_size=3*k)
    t0 = time()
    km_mini.fit(tfidf_matrix)
    print("done in %0.3fs" % (time() - t0))
    print(km_mini.inertia_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(data_label, km_mini.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(data_label, km_mini.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(data_label, km_mini.labels_))
    in_cluster_dis.append(km_mini.inertia_)
plt.plot(in_cluster_dis,'b.-')
plt.show()

# <-----------Bag of words, count vectorizer---------->
n_features = 50000
n_components = 500
count_vect = HashingVectorizer(n_features=n_features,
                               stop_words='english', non_negative=True,
                               norm=None, binary=False)
X = count_vect.fit_transform(data_articles)
print(X.shape)
print()
svd = TruncatedSVD(n_components)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
print(X.shape)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

#Hirarchical clustering - cosine distance
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(X[:400])

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=data_label[:400]);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

#search
in_cluster_dis = []
for k in xrange(1450, 2200, 50):
    print("")
    km_mini = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, init_size=3*k)
    t0 = time()
    km_mini.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print(km_mini.inertia_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(data_label, km_mini.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(data_label, km_mini.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(data_label, km_mini.labels_))
    in_cluster_dis.append(km_mini.inertia_)
