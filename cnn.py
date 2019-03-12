import numpy as np
import tensorflow as tf
import keras
from keras.layers.merge import Concatenate
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention


class TextCNNRNN(object):
    def printShapeArgs(self, *args):
        for arg in args:
            print("%s.shape : %s" % (arg, eval(arg)))

    def weight_variable(self, shape,tf_name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial,name=tf_name)

    def bias_variable(self, shape,tf_name):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial,name=tf_name)

    def conv2D(self,input, W, b,tf_name):
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID', name=tf_name)
        return tf.nn.relu(tf.nn.bias_add(conv, b))

    def __init__(self, non_static, hidden_unit, sequence_length, max_pool_size,embedding_mat,embedding_category_mat,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,alphabet_size= 70,attention_size = 100):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')
        # self.idk = tf.placeholder(tf.float32,[None,num_filters*len(filter_sizes)],name='idk')


        l2_loss = tf.constant(0.0)

        # categ_W = tf.Variable(embedding_category_mat, name="category_emb_W",dtype=tf.float32)
        # print("{} : {}".format("categW.shape", np.shape(categ_W)))
        print("{} : {}".format("embedding_mat.shape", np.shape(embedding_mat)))
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
            print("{} : {}".format("input_x.shape", np.shape(self.input_x)))
            print("{} : {}".format("embedded_chars.shape", np.shape(self.embedded_chars)))

            emb = tf.expand_dims(self.embedded_chars, -1)
        print("{} : {}".format("emb.shape", np.shape(emb)))

        # with tf.device('/cpu:0'), tf.name_scope('embedding'):
        #     # Quantization layer
        #
        #     Q = tf.concat(
        #         [
        #             tf.zeros([1, alphabet_size]),  # Zero padding vector for out of alphabet characters
        #             tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0)  # one-hot vector representation for alphabets
        #         ], 0,
        #         name='Q')
        #     self.embedded_chars = tf.nn.embedding_lookup(Q, self.input_x)
        #     emb = tf.expand_dims(self.embedded_chars, -1)  # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]

        self.pooled_concat = []
        for filter_size in filter_sizes:
            with tf.name_scope('conv-maxpool-%d' % int(3)):
                 # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                 W1 = self.weight_variable([int(filter_size),embedding_size, 1, num_filters],'W1')
                 b1 = self.bias_variable([num_filters],'b1')
                 conv1 = self.conv2D(emb,W1,b1,'Conv1')

                 pooled= tf.nn.max_pool(conv1, ksize=[1, sequence_length - int(filter_size) + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool2')

                 self.pooled_concat.append(pooled)

        self.pooled_concat = tf.concat(self.pooled_concat, 3)
        print("{} : {}".format("pooled_concat.shape", np.shape(self.pooled_concat)))
        self.pooled_concat = tf.reshape(self.pooled_concat, [-1, num_filters*len(filter_sizes)])

        # asd = tf.nn.embedding_lookup(categ_W,self.idk)
        # print("{} : {}".format("asd.shape", np.shape(asd)))
        # asd = tf.reshape(asd,[-1,num_filters*len(filter_sizes)])
        # print("{} : {}".format("asd_flatten.shape", np.shape(asd)))

        # pooled_concat = tf.concat([pooled_concat,self.idk],0)
        # print("{} : {}".format("concat.shape", np.shape(pooled_concat)))
        # pooled_concat = tf.layers.flatten(pooled_concat)
        self.pooled_concat = tf.nn.dropout(self.pooled_concat, self.dropout_keep_prob)
        print("{} : {}".format("pooled_concat_reshape.shape", np.shape(self.pooled_concat)))

        # W = tf.Variable(tf.truncated_normal([,num_filters*len(filter_sizes)]))

        # dense = tf.layers.dense(inputs=pooled_concat,units=num_filters*len(filter_sizes),activation=tf.nn.relu)
        # print("{} : {}".format("pooled_concat_reshape.shape", np.shape(pooled_concat)))
        # dense = tf.nn.dropout(dense,self.dropout_keep_prob)

        # dense1 = tf.layersde


        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([num_filters*len(filter_sizes), num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0., shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.pooled_concat, W, b, name='scores')
            print("{} : {}".format("scores.shape", self.scores))

            self.predictions = tf.argmax(self.scores, 1, name='predictions')
            print("{} : {}".format("predictions.shape", self.predictions))

        with tf.name_scope('loss'):
            print("{} : {}".format("scores.shape", self.scores))
            print("{} : {}".format("input_y.shape", self.input_y))
            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores))  # only named arguments accepted
            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name='accuracy')

        with tf.name_scope('num_correct'):
            self.correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(self.correct, 'float'))