import numpy as np
import tensorflow as tf
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
        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME', name=tf_name)
        return tf.nn.relu(tf.nn.bias_add(conv, b))

    def __init__(self, non_static, hidden_unit, sequence_length, max_pool_size,embedding_mat,
                 num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,alphabet_size= 70,attention_size = 100):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        l2_loss = tf.constant(0.0)

        # with tf.device('/gpu:0'), tf.name_scope('embedding'):
        #     if not non_static:
        #         W = tf.constant(embedding_mat, name='W')
        #     else:
        #         W = tf.Variable(embedding_mat, name='W')
        #     # W = tf.get_variable("embeddings",[alphabet_size,])
        #     # print("{} : {}".format("W.shape", np.shape(W)))
        #     # print(W)
        #     # print("{} : {}".format("input_x.shape", np.shape(self.input_x)))
        #     # print(self.input_x)
        #     self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        #
        #
        #     emb = tf.expand_dims(self.embedded_chars, -1)
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # Quantization layer

            Q = tf.concat(
                [
                    tf.zeros([1, alphabet_size]),  # Zero padding vector for out of alphabet characters
                    tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0)  # one-hot vector representation for alphabets
                ], 0,
                name='Q')
            self.embedded_chars = tf.nn.embedding_lookup(Q, self.input_x)
            emb = tf.expand_dims(self.embedded_chars, -1)  # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]

        pooled_concat = []
        for i in range(4):
            with tf.name_scope('conv-maxpool-%d' % int(3)):
                 # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                 W1 = self.weight_variable([5,sequence_length,1,64],'W1')
                 b1 = self.bias_variable([64],'b1')
                 conv1 = self.conv2D(emb,W1,b1,'Conv1')

                 pool2= tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

                 W3 = self.weight_variable([2,sequence_length,64,128],'W3')
                 b3 = self.bias_variable([128],'b3')
                 conv3 = self.conv2D(pool2,W3,b3,'Conv3')

                 # W4 = tf.Variable(tf.truncated_normal([3,sequence_length,128,256], stddev=0.1), name='W4')
                 # b4 = tf.Variable(tf.constant(0.1, shape=[256]), name='b4')
                 # conv4 = tf.nn.conv2d(conv3, W4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
                 # conv4 =  tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')
                 # print("{} : {}".format("conv4",np.shape(conv4)))
                 #
                 #
                 # W5 = tf.Variable(tf.truncated_normal([3,sequence_length,256,512], stddev=0.1), name='W5')
                 # b5 = tf.Variable(tf.constant(0.1, shape=[512]), name='b5')
                 # conv5 = tf.nn.conv2d(conv4, W5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
                 # conv5 =  tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
                 # print("{} : {}".format("conv5",np.shape(conv5)))
                 pooled = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name='pool6')
                 print("{} : {}".format("conv1", np.shape(conv1)))
                 print("{} : {}".format("pool2", np.shape(pool2)))
                 print("{} : {}".format("conv3", np.shape(conv3)))

                 print("{} : {}".format("pooled", np.shape(pooled)))



                 pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 3)
        print("{} : {}".format("pooled_concat.shape", np.shape(pooled_concat)))
        pooled_concat = tf.reshape(pooled_concat, [-1, sequence_length, 512 * 2])
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)
        print("{} : {}".format("pooled_concat_reshape.shape", np.shape(pooled_concat)))

        lstm_cell_fw = tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        lstm_cell_bw = tf.contrib.rnn.GRUCell(num_units=hidden_unit)

        outputs, state = bi_rnn(tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,self.dropout_keep_prob),
                                     tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,self.dropout_keep_prob),
                                     inputs=pooled_concat,
                                     dtype=tf.float32)

        # Attention
        with tf.name_scope('Attention_Layer'):
            attention_output, alphas, _ = attention(outputs, attention_size,time_major=True,return_alphas=True)

        #Dropout
        drop = tf.nn.dropout(attention_output,self.dropout_keep_prob)
        print("{} : {}".format("drop.shape", drop))

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([hidden_unit * 2, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0., shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(drop, W, b, name='scores')
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