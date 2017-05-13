import json
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

class wikinews_reader:
    def __init__(self, max_doc_length, max_sen_length, word_vocab, batch_size):
        with open('./data/wikinews.json', encoding="utf8") as data_file:  
            data = json.load(data_file)
        word_tokens = []
        full_articles = []
        titles = []
        for event in data:
            titles.append(event["title"])
            word_doc = []
            word_true = []
            for i in range(3):
                article = event["articles"][i]["body"].replace("\n\n", " ")
                sents = sent_tokenize(article)
                word_true +=sents
                for sent in sents:
                    sent = sent.strip()
                    sent= sent.replace('}', '').replace('{', '').replace('|', '')
                    sent = word_tokenize(sent)
                    if len(sent) > max_sen_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sen_length-2]
                    word_array = [word_vocab.get(c) for c in ['{'] + sent + ['}'] if word_vocab.get(c) is not None]
                    word_doc.append(word_array)           
            if len(word_doc) > max_doc_length:
                word_doc = word_doc[:max_doc_length]
            word_tokens.append(word_doc)
            full_articles.append(word_true)
        wiki_tensors = np.zeros([len(data), max_doc_length, max_sen_length], dtype=np.int32)
        for i, word_doc in enumerate(word_tokens):
            for j, word_array in enumerate(word_doc):
                wiki_tensors[i][j][0:len(word_array)] = word_array
        self.wiki_tensors = wiki_tensors
        self.ori = full_articles
        self.titles = titles
        self.pointer = 0
        self.N = wiki_tensors.shape[0]
        self.batch_size = batch_size
        self.batch_num = int(np.ceil(float(self.N)/self.batch_size))
    def next_batch(self):
        if self.batch_size+self.pointer < self.N:
            self.pointer += self.batch_size
            return (self.wiki_tensors[self.pointer-self.batch_size:self.pointer], 
                   self.ori[self.pointer-self.batch_size:self.pointer], 
                   self.titles[self.pointer-self.batch_size:self.pointer])
        else:
            left_num = self.batch_size - (self.N - self.pointer)
            w = np.concatenate([self.wiki_tensors[-1:]]*left_num, axis=0)
            w = np.concatenate([self.wiki_tensors[self.pointer:], w], axis=0)
            o = self.ori[self.pointer:] + self.ori[-1:]*left_num
            t = self.titles[self.pointer:] + self.titles[-1:]*left_num
            return w, o, t

class news_reader:
    def __init__(self, max_doc_length, max_sen_length, word_vocab, batch_size, events):
        word_tokens = []
        full_articles = []
        for event in events:
            word_doc = []
            word_true = []
            for i in range(2):
                article = event[i].replace("\n\n", " ")
                sents = sent_tokenize(article)
                word_true +=sents
                for sent in sents:
                    sent = sent.strip()
                    sent= sent.replace('}', '').replace('{', '').replace('|', '')
                    sent = word_tokenize(sent)
                    if len(sent) > max_sen_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sen_length-2]
                    word_array = [word_vocab.get(c) for c in ['{'] + sent + ['}'] if word_vocab.get(c) is not None]
                    word_doc.append(word_array)           
            if len(word_doc) > max_doc_length:
                word_doc = word_doc[:max_doc_length]
            word_tokens.append(word_doc)
            full_articles.append(word_true)
        wiki_tensors = np.zeros([len(events), max_doc_length, max_sen_length], dtype=np.int32)
        for i, word_doc in enumerate(word_tokens):
            for j, word_array in enumerate(word_doc):
                wiki_tensors[i][j][0:len(word_array)] = word_array
        self.wiki_tensors = wiki_tensors
        self.ori = full_articles
        self.pointer = 0
        self.N = wiki_tensors.shape[0]
        self.batch_size = batch_size
        self.batch_num = int(np.ceil(float(self.N)/self.batch_size))
    def next_batch(self):
        if self.batch_size+self.pointer < self.N:
            self.pointer += self.batch_size
            return (self.wiki_tensors[self.pointer-self.batch_size:self.pointer], 
                   self.ori[self.pointer-self.batch_size:self.pointer])
        else:
            left_num = self.batch_size - (self.N - self.pointer)
            w = np.concatenate([self.wiki_tensors[-1:]]*left_num, axis=0)
            w = np.concatenate([self.wiki_tensors[self.pointer:], w], axis=0)
            o = self.ori[self.pointer:] + self.ori[-1:]*left_num
            return w, o



