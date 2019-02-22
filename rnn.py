import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention


class TextCNNRNN(object):
    def __init__(self, non_static, hidden_unit, sequence_length, max_pool_size,embedding_mat,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,alphabet_size= 70,attention_size = 100):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            if not non_static:
                W = tf.constant(embedding_mat, name='W')
            else:
                W = tf.Variable(embedding_mat, name='W')
            # W = tf.get_variable("embeddings",[alphabet_size,])
            # print("{} : {}".format("W.shape", np.shape(W)))
            # print(W)
            # print("{} : {}".format("input_x.shape", np.shape(self.input_x)))
            # print(self.input_x)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)


        rnnCell = tf.nn.rnn_cell.LSTMCell(hidden_unit,activation=tf.nn.relu)
        outputs, _ = tf.nn.dynamic_rnn(rnnCell,self.embedded_chars, dtype=tf.float32)

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0., shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(outputs[:,-1], W, b, name='scores')
            print("{} : {}".format("scores.shape", self.scores))

            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            print("{} : {}".format("scores.shape", self.scores))
            print("{} : {}".format("input_y.shape", self.input_y))
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))  # only named arguments accepted
            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, 'float'))