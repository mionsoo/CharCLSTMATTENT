import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from attention import attention
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.regularizers import l2


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

        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            # Quantization layer

            Q = tf.concat(
                [
                    tf.zeros([1, alphabet_size]),  # Zero padding vector for out of alphabet characters
                    tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0)  # one-hot vector representation for alphabets
                ], 0,
                name='Q')

            print("{} : {}".format("W.shape", np.shape(Q)))
            print()
            print("{} : {}".format("input_x.shape", np.shape(self.input_x)))
            self.embedded_chars = tf.nn.embedding_lookup(Q, self.input_x)
            print("{} : {}".format("self.embedded_chars.shape", self.embedded_chars))

            emb = tf.expand_dims(self.embedded_chars, -1)  # Add the channel dim, thus the shape of x is [batch_size, l0, alphabet_size, 1]

        print("{} : {}".format("emb.shape",emb))


        pooled_concat = []
        reduced = np.int32(np.ceil((sequence_length) * 1.0 / max_pool_size))
        print("{} : {}".format("reduced.shape", reduced))



        # i = 1
        # with tf.name_scope('conv-maxpool-%s'% i):
        #     # filter_shape = [filter_sizes[0][1],embedding_size,1,filter_sizes[0][0]]
        #     W1 = tf.Variable(tf.truncated_normal([filter_sizes[0][1],embedding_size,self.input_x.get_shape(),filter_sizes[0][0]],stddev=0.1),name='W1')
        #     b1 = tf.Variable(tf.constant(0.1,shape=[filter_sizes[0][0]]),name='b1')
        #     conv1 = tf.nn.conv2d(emb,W1,strides=[1,1,1,1],padding='VALID',name='conv1')
        #     conv1 = tf.nn.relu(tf.nn.bias_add(conv1,b1),name='relu1')
        #     print(conv1)
        #     pool2 = tf.nn.max_pool(conv1, ksize=[1,filter_sizes[1][1],filter_sizes[1][1],1],strides=[1,filter_sizes[1][1],filter_sizes[1][1],1],padding='VALID')
        #     # pool2 = tf.nn.dropout(pool2,keep_prob=self.dropout_keep_prob)
        #     print(pool2)
        #     W3 = tf.Variable(tf.truncated_normal([filter_sizes[2][1],embedding_size,filter_sizes[1][0],filter_sizes[2][0]],stddev=0.1),name='W3')
        #     b3 = tf.Variable(tf.constant(0.1,shape=[filter_sizes[2][0]]),name='b2')
        #     print(W3)
        #     conv3 = tf.nn.conv2d(pool2,W3,strides=[1,1,1,1],padding='VALID',name='conv3')
        #     conv3 = tf.nn.relu(tf.nn.bias_add(conv3,b3),name='relu3')
        #
        #     W4 = tf.Variable(tf.truncated_normal([filter_sizes[3][1], embedding_size, filter_sizes[2][0], filter_sizes[3][0]], stddev=0.1), name='W4')
        #     b4 = tf.Variable(tf.constant(0.1, shape=[filter_sizes[3][0]]), name='b4')
        #     conv4 = tf.nn.conv2d(conv3, W4, strides=[1, 1, 1, 1], padding='VALID', name='conv4')
        #     conv4 = tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')
        #
        #     W5 = tf.Variable(tf.truncated_normal([filter_sizes[4][1], embedding_size, filter_sizes[3][0], filter_sizes[4][0]], stddev=0.1), name='W5')
        #     b5 = tf.Variable(tf.constant(0.1, shape=[filter_sizes[4][0]]), name='b5')
        #     conv5 = tf.nn.conv2d(conv4, W5, strides=[1, 1, 1, 1], padding='VALID', name='conv5')
        #     conv5 = tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
        #
        #     pool6 = tf.nn.max_pool(conv5, ksize=[1, filter_sizes[5][1], 1, 1], strides=[1, filter_sizes[5][1], 1, 1], padding='VALID')
        #
        #     pooled = tf.reshape(pool6,[-1,reduced,num_filters])
        #     pooled_concat.append(pooled)


        # var_id = 0
        # i=1
        # with tf.name_scope('conv-maxpool-%s'%i):
        #     filter_size = 1
        #     num_filters = 0
        #
        #     # num_prio = (filter_sizes[] - 1) // 2
        #     # num_post = (filter_size - 1) - num_prio
        #     # pad_prio = tf.concat([self.pad] * num_prio, 1)
        #     # pad_post = tf.concat([self.pad] * num_post, 1)
        #     # emb_pad = tf.concat([pad_prio, emb, pad_post], 1)
        #     # print("{} : {}".format("emb_pad.shape", np.shape(emb_pad)))
        #
        #     # filter_shape = [filter_size,embedding_size,1,num_filters]
        #     conv1 = tf.layers.conv2d(inputs=emb,filters=64,kernel_size=[5,1],padding='VALID',activation=tf.nn.relu)
        #     print("{} : {}".format("conv1", conv1))
        #     pool2 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,1],padding="VALID", strides=1)
        #     print("{} : {}".format("pool2", pool2))
        #     conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[3,1],padding='VALID',activation=tf.nn.relu)
        #     print("{} : {}".format("conv3", conv3))
        #     conv4 = tf.layers.conv2d(inputs=conv3,filters=256,kernel_size=[3,1],padding='VALID',activation=tf.nn.relu)
        #     print("{} : {}".format("conv4", conv4))
        #     conv5 = tf.layers.conv2d(inputs=conv4,filters=256,kernel_size=[3,1],padding='VALID',activation=tf.nn.relu)
        #     print("{} : {}".format("conv5", conv5))
        #     pool6 = tf.layers.max_pooling2d(inputs=conv5,pool_size=[3,1],padding="VALID",strides=1)
        #     print("{} : {}".format("pool6", pool6))
        #
        #     pooled = tf.reshape(pool6,[-1,reduced,256])
        #     print("{} : {}".format("pooled", pooled))
        #     pooled_concat.append(pooled)


        # def conv2d(filter_size,embedding_size,num_filter,_input):
        #     filter_shape = [int(filter_size), embedding_size, 1, num_filter]
        #     W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        #     # print("{} : {}".format("conv_W.shape", np.shape(W)))
        #     b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name='b')
        #     # print("{} : {}".format("conv_b.shape", np.shape(b)))
        #     conv = tf.nn.conv2d(_input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        #     # print("{} : {}".format("conv.shape",np.shape(conv)))
        #     return tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        #
        # def conv2d2(filter_size,embedding_size,num_filter,_input,before_num_filter):
        #     filter_shape = [int(filter_size), embedding_size, before_num_filter, num_filter]
        #     W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        #     # print("{} : {}".format("conv_W.shape", np.shape(W)))
        #     b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name='b')
        #     # print("{} : {}".format("conv_b.shape", np.shape(b)))
        #     conv = tf.nn.conv2d(_input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        #     # print("{} : {}".format("conv.shape",np.shape(conv)))
        #     return tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        #
        #
        # def maxpool(max_pool_size,_input):
        #     return tf.nn.max_pool(_input, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1], padding='VALID', name='pool')





        # for i in range(4):
        with tf.name_scope('conv-maxpool-%d' % int(3)):
             # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
             # num_prio = (int(filter_size) - 1) // 2
             # num_post = (int(filter_size) - 1) - num_prio
             # pad_prio = tf.concat([self.pad] * num_prio, 1)
             # pad_post = tf.concat([self.pad] * num_post, 1)
             # emb_pad = tf.concat([pad_prio, emb, pad_post], 1)
             # print("{} : {}".format("emb_pad.shape", np.shape(emb_pad)))
             W1 = tf.Variable(tf.truncated_normal([5,sequence_length,1,64], stddev=0.1), name='W1')
             b1 = tf.Variable(tf.constant(0.1, shape=[64]), name='b1')
             conv1 = tf.nn.conv2d(emb, W1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
             conv1 =  tf.nn.relu(tf.nn.bias_add(conv1, b1), name='relu1')
             print("{} : {}".format("conv1",np.shape(conv1)))

             pool2= tf.nn.max_pool(conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME', name='pool2')
             # pool2 = tf.transpose(pool2,[0,1,3,2])
             print("{} : {}".format("pool2",np.shape(pool2)))

             W3 = tf.Variable(tf.truncated_normal([2,sequence_length,64,128], stddev=0.1), name='W3')
             b3 = tf.Variable(tf.constant(0.1, shape=[128]), name='b3')
             conv3 = tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
             conv3 =  tf.nn.relu(tf.nn.bias_add(conv3, b3), name='relu3')
             # conv3 = tf.transpose(conv3, [0, 1, 3, 2])
             print("{} : {}".format("conv3",np.shape(conv3)))

             W4 = tf.Variable(tf.truncated_normal([3,sequence_length,128,256], stddev=0.1), name='W4')
             b4 = tf.Variable(tf.constant(0.1, shape=[256]), name='b4')
             conv4 = tf.nn.conv2d(conv3, W4, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
             conv4 =  tf.nn.relu(tf.nn.bias_add(conv4, b4), name='relu4')
             # conv4 = tf.transpose(conv4, [0, 1, 3, 2])
             print("{} : {}".format("conv4",np.shape(conv4)))


             W5 = tf.Variable(tf.truncated_normal([3,sequence_length,256,512], stddev=0.1), name='W5')
             b5 = tf.Variable(tf.constant(0.1, shape=[512]), name='b5')
             conv5 = tf.nn.conv2d(conv4, W5, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
             conv5 =  tf.nn.relu(tf.nn.bias_add(conv5, b5), name='relu5')
             print("{} : {}".format("conv5",np.shape(conv5)))


             pooled = tf.nn.max_pool(conv5, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME', name='pool6')
             # pooled = tf.transpose(pooled, [0, 1, 3, 2])
             print("{} : {}".format("pooled", np.shape(pooled)))
             # conv1 = conv2d(filter_sizes[0],embedding_size,num_filters[0],emb)
             # print("{} : {}".format("conv1",np.shape(conv1)))
             # pooled2 = maxpool(filter_sizes[1],conv1)
             # print("{} : {}".format("pooled2",np.shape(pooled2)))
             # conv3 = conv2d2(filter_sizes[2],embedding_size,num_filters[2],pooled2,num_filters[0])
             # conv3 = tf.reshape(conv3,[0,1,3,2])
             # conv4 = conv2d(filter_sizes[3],embedding_size,num_filters[3],conv3)
             # conv4 = tf.reshape(conv4,[0,1,3,2])
             # conv5 = conv2d(filter_sizes[4],embedding_size,num_filters[4],conv4)
             # pooled = maxpool(filter_sizes[5],conv5)

             # Maxpooling over the outputs
             # pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1], padding='SAME', name='pool')
             # print("{} : {}".format("pooled.shape",np.shape(pooled)))
             pooled = tf.reshape(pooled, [-1,reduced,512])
             print("{} : {}".format("pooled_reshape.shape",np.shape(pooled)))
             pooled_concat.append(pooled)

        '''
        for i in range(4):
            with tf.name_scope('conv-maxpool-%d' % int(i)):
                for filter_size in filter_sizes:
                    # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                    num_prio = (int(filter_size) - 1) // 2
                    num_post = (int(filter_size) - 1) - num_prio
                    pad_prio = tf.concat([self.pad] * num_prio, 1)
                    pad_post = tf.concat([self.pad] * num_post, 1)
                    emb_pad = tf.concat([pad_prio, emb, pad_post], 1)
                    # print("{} : {}".format("emb_pad.shape", np.shape(emb_pad)))

                    filter_shape = [int(filter_size), embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    # print("{} : {}".format("conv_W.shape", np.shape(W)))
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    # print("{} : {}".format("conv_b.shape", np.shape(b)))
                    conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    # print("{} : {}".format("conv.shape",np.shape(conv)))

                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # print("{} : {}".format("conv_h.shape",np.shape(h)))

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, max_pool_size, 1, 1], strides=[1, max_pool_size, 1, 1], padding='SAME', name='pool')
                    # print("{} : {}".format("pooled.shape",np.shape(pooled)))
                    pooled = tf.reshape(pooled, [-1, reduced, num_filters])
                    # print("{} : {}".format("pooled_reshape.shape",np.shape(pooled)))
                    pooled_concat.append(pooled)
        '''

        pooled_concat = tf.concat(pooled_concat, 2)

        # print("{} : {}".format("pooled_concat.shape",np.shape(pooled_concat)))
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)
        # print("{} : {}".format("pooled_concat_dropout.shape",np.shape(pooled_concat)))

        lstm_cell = tf.contrib.rnn.GRUCell(num_units=hidden_unit)

        # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
        # lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)

        self._initial_state = lstm_cell.zero_state(self.batch_size, tf.float32)
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(pooled_concat, num_or_size_splits=int(reduced), axis=1)]
        # outputs, state = tf.nn.rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)
        # print("{} : {}".format("squeezePooled_concat_input.shape",np.shape(pooled_concat)))
        # print(pooled_concat)
        # outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, inputs, initial_state=self._initial_state, sequence_length=self.real_len)
        outputs, state = bi_rnn(tf.nn.rnn_cell.DropoutWrapper(lstm_cell,self.dropout_keep_prob),
                                     tf.nn.rnn_cell.DropoutWrapper(lstm_cell,self.dropout_keep_prob),
                                     inputs=pooled_concat,
                                     dtype=tf.float32)


        # print(outputs)
        # print("{} : {}".format("outputs.shape",np.shape(outputs)))


        # Attention
        with tf.name_scope('Attention_Layer'):
            attention_output, alphas, _ = attention(outputs, attention_size,return_alphas=True)

        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)
        # output = outputs[0]
        # with tf.variable_scope('Output'):
        #     tf.get_variable_scope().reuse_variables()
        #     one = tf.ones([1, hidden_unit], tf.float32)
        #     for i in range(1, len(outputs)):
        #         ind = self.real_len < (i + 1)
        #         ind = tf.to_float(ind)
        #         ind = tf.expand_dims(ind, -1)
        #         mat = tf.matmul(ind, one)
        #         output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))
        #         print("{} : {}".format("output.shape",np.shape(output)))

        #Dropout
        drop = tf.nn.dropout(attention_output,self.dropout_keep_prob)
        print("{} : {}".format("drop.shape", drop))

        # drop = tf.reshape(drop, [-1, (hidden_unit * max_pool_size)*4])

        # print("{} : {}".format("score_input.shape",np.shape(outputs)))
        # print("{} : {}".format("pad.shape",np.shape(self.pad)))
        # print(self.pad)

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