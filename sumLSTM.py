from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import pickle
 
from sklearn.metrics import precision_recall_fscore_support as score
from reader import wikinews_reader, load_data, DataReader
from utils import conv2d, linear, load_wordvec, cnn


class Trainer(object):
    def __init__(self, 
                  initializer, 
                  word_vocab,
                  embedding_path,
                  word_embed_size = 100,
                  max_doc_length = 15,
                  max_sen_length = 50,
                  kernels = [1, 2, 3, 4, 5, 6, 7],
                  kernel_features = [50, 100, 150, 200, 200, 200, 200],
                  num_rnn_layers = 2,
                  rnn_size = 650,
                  dropout = 0.5,
                  learning_rate = 1.0,
                  max_grad_norm = 5.0,
                  batch_size = 20,
                  ):
        self._initializer = initializer
        self._word_vocab = word_vocab
        self._embedding_path = embedding_path
        self._word_embed_size = word_embed_size
        self._max_doc_length = max_doc_length
        self._max_sen_length = max_sen_length
        self._kernels = kernels
        self._kernel_features = kernel_features
        self._num_rnn_layers = num_rnn_layers
        self._rnn_size = rnn_size
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._max_grad_norm = max_grad_norm
        self._batch_size = batch_size
        self._model()    
    def _model(self):
        with tf.variable_scope("Model", initializer=self._initializer):
            # Input sentence [Batch_size, timestep]
            pretrained_emb = load_wordvec(self._embedding_path, self._word_vocab, self._word_embed_size)
            self.input = tf.placeholder(tf.int32, shape=[self._batch_size, self._max_doc_length, self._max_sen_length], name="input")
            word_embedding = tf.get_variable(name='word_embedding4', shape=[self._word_vocab.size, self._word_embed_size], 
                                       initializer=tf.constant_initializer(pretrained_emb))
            self.clear_word_embedding_padding = tf.scatter_update(word_embedding, [0], tf.constant(0.0, shape=[1, self._word_embed_size]))
            self.input_embedded = tf.nn.embedding_lookup(word_embedding, self.input)
            self.input_embedded = tf.reshape(self.input_embedded, [-1, self._max_sen_length, self._word_embed_size])
            # CNN Encoder of embeddings
            # [batch_size x max_doc_length, cnn_size]
            self.input_cnn = cnn(self.input_embedded, self._kernels, self._kernel_features)
            self.input_cnn = tf.reshape(self.input_cnn, [self._batch_size, self._max_doc_length, -1])
            self.input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(self.input_cnn, self._max_doc_length, 1)]
            # LSTM Encoder
            with tf.variable_scope('LSTMencoder'):
                cell = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell() for _ in range(self._num_rnn_layers)], state_is_tuple=True)
                self.initial_enc_state = cell.zero_state(self._batch_size, dtype=tf.float32)
                # LSTM Encoder Run
                self.enc_outputs, self.final_enc_state = tf.contrib.rnn.static_rnn(cell, self.input_cnn2,
                                             initial_state=self.initial_enc_state, dtype=tf.float32)
            # LSTM Decoder
            with tf.variable_scope('LSTMdecoder'):
                cell2 = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell() for _ in range(self._num_rnn_layers)], state_is_tuple=True)
                self.initial_dec_state = self.final_enc_state
                self.dec_outputs, self.final_dec_state = tf.contrib.rnn.static_rnn(cell2, self.input_cnn2,
                                         initial_state=self.initial_dec_state, dtype=tf.float32)
            # Predictions
            self.logits = []
            with tf.variable_scope('Prediction') as scope:
                for idx, output in enumerate(zip(self.enc_outputs, self.dec_outputs)):
                    if idx > 0:
                        scope.reuse_variables()
                    output_enc, output_dec = output
                    self.logits.append(linear(tf.concat([output_enc, output_dec], 1), 3))
            # Loss Function
            with tf.variable_scope('Loss'):
                self.targets = tf.placeholder(tf.int64, [self._batch_size, self._max_doc_length], name='targets')
                target_list = [tf.squeeze(x, [1]) for x in tf.split(self.targets, self._max_doc_length, 1)]
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = target_list), name='loss')
            # Training
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope('SGD_Training'):
                self.learning_rate = tf.Variable(self._learning_rate, trainable=False, name='learning_rate')
                tvars = tf.trainable_variables()
                grads, self.global_norm = tf.clip_by_global_norm(tf.gradients(self.loss * self._max_doc_length, tvars), self._max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    def create_rnn_cell(self):
        """
        Create an rnn_cell with dropout
        """
        cell = tf.contrib.rnn.BasicLSTMCell(self._rnn_size, state_is_tuple=True, forget_bias=0.0)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-self._dropout)
        return cell
    def _create_session(self):
        # Config for Euler and GPUs
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   
    def train(self, train_reader, valid_reader, restore):
        saver = tf.train.Saver(max_to_keep=50)
        # Launch the graph
        min_val_loss = None
        last_improve_ep = 0
        start_time = time.time()
        with self._create_session() as sess:
            if restore:
                saver.restore(session, "./model.ckpt")
                print('Loaded model at global step', self.global_step.eval())
            else:
                init = tf.global_variables_initializer()
                sess.run(init)
                sess.run(self.clear_word_embedding_padding)
                print("Initializing variables...")           
            print("Running Model")
            for epoch in range(50):
                avg_cost = 0.
                report_time = time.time()
                count = 0
                for x, y in train_reader.iter():
                    count += 1
                    loss, _, gradient_norm, step, _ = sess.run([
                        self.loss,
                        self.train_op,
                        self.global_norm,
                        self.global_step,
                        self.clear_word_embedding_padding
                        ], {
                            self.input  : x,
                            self.targets: y,
                        })
                    avg_cost += 0.05 * (loss - avg_cost)
                    if count%5 == 0:
                        print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f grad.norm=%6.8f' % (step,
                                                            epoch, count,
                                                            train_reader.length,
                                                            loss, np.exp(loss),
                                                            gradient_norm))
                print('Epoch training time:', time.time()-report_time)               
                val_loss = 0.0
                count = 0
                for x, y in valid_reader.iter():
                    count += 1
                    loss = sess.run(self.loss, {self.input: x, self.targets: y,})
                    val_loss += loss / valid_reader.length
                print("at the end of epoch:", epoch)
                print("train loss = %6.8f, perplexity = %6.8f" % (avg_cost, np.exp(avg_cost)))
                print("validation loss = %6.8f, perplexity = %6.8f" % (val_loss, np.exp(val_loss)))
                # Save the model if it achieved better result on val
                if min_val_loss==None or val_loss<min_val_loss:
                    model_path = saver.save(sess, "./model.ckpt")
                    min_val_loss = val_loss
                    last_improve_ep = epoch
                    save_message = ", Model saved!".format(model_path)  
                    print(save_message)             
                # Early termination if no improvement is found for x epochs.
                if epoch - last_improve_ep > 10:
                    print("No improvement during the past {} epochs, stopping optimization".format(last_improve_ep))
                    break
                # lr decay
                if min_val_loss != None and np.exp(val_loss) > np.exp(min_val_loss) - 1.0:
                    current_lr = sess.run(self.learning_rate) * 0.5
                    if current_lr < 1.e-5:
                        print('Finish training')
                        break
                    sess.run(self.learning_rate.assign(current_lr))
                    print('New learning rate is:', current_lr)
            elapsed_time = time.time() - start_time
            elapsed_time_message = "Total training time: {:.3f} sec".format(elapsed_time)
            print(elapsed_time_message)
   
    def test(self, test_reader):
        saver = tf.train.Saver()
        with self._create_session() as sess:
            # Restore trained model
            saver.restore(sess, "./model.ckpt")
            print("Model loaded")
            # Predictions
            batch_num = len(list(test_reader.iter()))
            precision, recall, fscore = 0., 0., 0.
            for x, y in test_reader.iter():
                logits = sess.run([self.logits],{self.input: x})
                logits = np.array(logits[0])
                pred = np.argmax(np.transpose(logits, (1,0,2)), axis=2)
                for i in range(self._batch_size):
                    predi = [0 if e==2 else e for e in pred[i]]
                    yi = [0 if e==2 else e for e in y[i]]
                    precision_, recall_, fscore_, _ = score(predi, yi, labels=[0,1])
                    precision += precision_[1]
                    recall += recall_[1]
                    fscore += fscore_[1]           
            precision /= batch_num * self._batch_size
            recall /= batch_num * self._batch_size
            fscore /= batch_num * self._batch_size
            print("precision:", precision, "recall:", recall, "fscore:", fscore)
    def gen(self, wr):
        saver = tf.train.Saver()
        with self._create_session() as sess:
            # Restore trained model
            saver.restore(sess, "./model.ckpt")
            print("Model loaded")
            # Summary files
            save_file = open("./summary/summary.txt", "w")
            # Predictions
            for i in range(wr.batch_num):
                x, o, t = wr.next_batch()
                logits = sess.run([self.logits],{self.input: x})
                logits = np.array(logits[0])
                pred = np.argmax(np.transpose(logits, (1,0,2)), axis=2)
                for j in range(self._batch_size):
                    summary = ""
                    for k, label in enumerate(pred[j]):
                        if label==1:
                            summary += o[j][k] + " "
                    save_file.write('=================================\n')
                    save_file.write(t[j]+'\n')
                    save_file.write(summary+ '\n\n')
            save_file.close()
            

    

if __name__ == "__main__":
    max_doc_length = 15
    max_sen_length = 50
    batch_size = 20

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--restore', action='store_true', 
                        help='restore model')
    parser.add_argument('--train', action='store_true', 
                        help='train model')
    parser.add_argument('--eval', action='store_true', 
                        help='restore model')
    parser.add_argument('--generate', action='store_true', 
                        help='restore model')
    args = parser.parse_args()

    embedding_path = "./data/wordembeddings-dim100.word2vec"

    if args.train or args.test:
        word_vocab, word_tensors, max_doc_length, label_tensors = \
            load_data('data/', max_doc_length, max_sen_length)

        train_reader = DataReader(word_tensors['train'], label_tensors['train'],
                                  batch_size)

        valid_reader = DataReader(word_tensors['valid'], label_tensors['valid'],
                                  batch_size)

        test_reader = DataReader(word_tensors['test'], label_tensors['test'],
                                  batch_size)
        with open('./data/word_vocab.pkl', 'wb') as output:
            pickle.dump(word_vocab, output, pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open('./data/word_vocab.pkl', 'rb') as ipt:
                word_vocab = pickle.load(ipt)
        except:
            pass

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    trainer = Trainer(initializer, word_vocab, embedding_path,
                      max_doc_length = max_doc_length,
                      max_sen_length = max_sen_length,
                      batch_size = batch_size)
    print("Trainer established")
    if args.train:
        trainer.train(train_reader, valid_reader, args.restore)
    if args.eval:
        print("Evaluating")
        trainer.test(test_reader)
    if args.generate:
        print("Testing on WikiNews")
        wiki_reader = wikinews_reader(max_doc_length, max_sen_length, word_vocab, batch_size)
        trainer.gen(wiki_reader)
    
