from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import json
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



class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_doc_length=10, max_sent_length=50):

    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)
        for dname in os.listdir(pname):  

            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                lines = f.read().split('\n\n')
                word_doc = []
                label_doc = []

                for line in lines[1].split('\n'):
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')

                    sent, label = line.split('\t\t\t')  
                    label_doc.append(label)
                    sent = sent.split(' ')

                    if len(sent) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sent_length-2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent + ['}']]
                    word_doc.append(word_array)
  
                if len(word_doc) > max_doc_length:
                    word_doc = word_doc[:max_doc_length]
                    label_doc = label_doc[:max_doc_length]

                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)
                labels[fname].append(label_doc)

    assert actual_max_doc_length <= max_doc_length

    print()
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length], dtype=np.int32)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int32)
 
        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors


class DataReader:

    def __init__(self, word_tensor, label_tensor, batch_size):

        length = word_tensor.shape[0]

        doc_length = word_tensor.shape[1]
        sent_length = word_tensor.shape[2]

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


def load_data_abs(data_dir, max_doc_length=10, max_sent_length=50, max_output_length=100, use_abs=True):
    '''
        data loader for generation models
        use_abs: When it is set to True, we use the human summaries as target;
                 otherwise we use the sentences labeled with 1 as target.
    '''

    word_vocab = Vocab()
    word_vocab.feed(' ')
    word_vocab.feed('{')
    word_vocab.feed('}')

    abs_vocab = Vocab()
    abs_vocab.feed(' ')
    abs_vocab.feed('{')
    abs_vocab.feed('}')

    actual_max_doc_length = 0
    actual_max_ext_length = 0
    actual_max_abs_length = 0

    word_tokens = collections.defaultdict(list)
    ext_output = collections.defaultdict(list)
    abs_output = collections.defaultdict(list)

    for fname in ('train', 'valid', 'test'):
        print('reading', fname)
        pname = os.path.join(data_dir, fname)
        for dname in os.listdir(pname):  

            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                lines = f.read().split('\n\n')
                word_doc = []
                ext_doc = []

                for line in lines[1].split('\n'):
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')

                    sent, label = line.split('\t\t\t')  
                    sent = sent.split(' ')

                    if len(sent) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sent_length-2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent + ['}']]

                    word_doc.append(word_array)
                 
                    if label == '1':
                        ext_doc.extend(word_array[1:-1])
 
                    if len(word_doc) == max_doc_length:
                        break

                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)

                if len(ext_doc) > max_output_length - 2:
                    ext_doc = ext_doc[:max_output_length-2]

                ext_doc = [word_vocab['{']] + ext_doc + [word_vocab['}']]
                ext_output[fname].append(ext_doc)
     
                actual_max_ext_length = max(actual_max_ext_length, len(ext_doc))

                abs_doc = lines[2].replace('\n', ' ')
                abs_doc = abs_doc.split(' ')
                if len(abs_doc) > max_output_length - 2:
                    abs_doc = abs_doc[:max_output_length-2]

                abs_doc = [abs_vocab.feed(c) for c in ['{'] + abs_doc + ['}']]       
                abs_output[fname].append(abs_doc)
 
                actual_max_abs_length = max(actual_max_abs_length, len(abs_doc))

    assert actual_max_doc_length <= max_doc_length

    print()
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    target_tensors = {}
    target_vocab = word_vocab
    actual_max_target_length = actual_max_ext_length

    if use_abs:
        target_vocab = abs_vocab
        actual_max_target_length = actual_max_abs_length

    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length], dtype=np.int32)
        target_tensors[fname] = np.zeros([len(ext_output[fname]), max_output_length], dtype=np.int32)
 
        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        if use_abs:
            for i, abs_doc in enumerate(abs_output[fname]):
                target_tensors[fname][i][0:len(abs_doc)] = abs_doc
        else:
            for i, ext_doc in enumerate(ext_output[fname]):
                target_tensors[fname][i][0:len(ext_doc)] = ext_doc

    return word_vocab, word_tensors, actual_max_doc_length, target_vocab, target_tensors, actual_max_target_length


class DataReader_abs:

    def __init__(self, word_tensor, target_tensor, batch_size):

        length = word_tensor.shape[0]

        doc_length = word_tensor.shape[1]
        sent_length = word_tensor.shape[2]

        output_length = target_tensor.shape[1]
        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        word_tensor = word_tensor[:clipped_length]
        target_tensor = target_tensor[:clipped_length]

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        y_batches = target_tensor.reshape([batch_size, -1, output_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):

        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y
