import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from functools import reduce
from operator import itemgetter
from utils import preprocess, flatten
import itertools


def probability_distribution(tokens):
    N, distinct_w = len(tokens), set(tokens)      
    p = [tokens.count(w) / float(N) for w in distinct_w]
    return dict(zip(distinct_w, p))

def sentence_weight(sentence, distribution):
    tokens = preprocess((word_tokenize(sentence)))
    return reduce(lambda x,y: x+y, [distribution[x] for x in tokens]) / len(tokens)

def max_weight_sentence(sentences, weights):
    return max(zip(sentences, weights), key=itemgetter(1))[0]


def sum_basic(lines, word_limit):    
    sentences = flatten([sent_tokenize(line) for line in lines])
    tokens = preprocess(flatten([word_tokenize(sent) for sent in sentences]))
    pd = probability_distribution(tokens)

    summary = "" 
    while len(word_tokenize(summary)) < word_limit:
        weights = [sentence_weight(sentence, pd) for sentence in sentences]
        highest_weight_sentence = max_weight_sentence(sentences, weights)
        summary += " " + highest_weight_sentence
        for token in preprocess(word_tokenize(highest_weight_sentence)):
            pd[token] = pd[token] * pd[token]

    return summary 

    
    