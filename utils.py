import tensorflow as tf 
from nltk.corpus import stopwords
import itertools
import numpy as np

"""
A collections of functions, some from NeuralSum project
"""

def preprocess(tokens):
    stop_words = set(stopwords.words('english'))
    l_tokens = [t.lower() for t in tokens]
    return [t for t in l_tokens if t not in stop_words]

def flatten(l):
    return list(itertools.chain.from_iterable(l))


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])
    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    '''
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]
    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term



def load_wordvec(embedding_path, word_vocab, word_embed_size):
    '''
    loads pretrained word vectors
    '''
    initW = np.random.uniform(-0.25, 0.25, (word_vocab.size, word_embed_size))
    with open(embedding_path, "r") as f:
        for line in f:
            line = line.rstrip().split(' ')
            word, vec = line[0], line[1:]
            if word in word_vocab.token2index.keys():
                initW[word_vocab[word]] = np.asarray([float(x) for x in vec]) 
    return initW

def cnn(input_, kernels, kernel_features, scope='CNN1'):
    '''
    CNN operation on embedded input
    Args:
        input:           input float tensor of shape [(batch_size*max_doc_length) x max_sen_length x embed_size]
        kernels:         array of kernel sizes
        kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    max_sen_length, embed_size = input_.get_shape()[1], input_.get_shape()[-1]
    # [batch_size*max_doc_length, 1, max_sen_length, embed_size]
    input_ = tf.expand_dims(input_, 1)
    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_sen_length - kernel_size + 1
            # [batch_size x max_sen_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)
            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(pool, [1, 2]))
        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
    return output