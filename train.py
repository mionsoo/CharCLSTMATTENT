import os
import json
import time
import math
import pickle
import logging
import data_helper
import datetime
import numpy as np
import tensorflow as tf
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from itertools import product
from multiprocessing import Pool
from math import log
import keras
import tqdm

# CNN
from cnn import TextCNNRNN

# RNN
# from rnn import TextCNNRNN

# CRNN
# from char_cnn_biLstm2 import TextCNNRNN



logging.getLogger().setLevel(logging.INFO)


tf.flags.DEFINE_string("filename", "cnn_root_Accuracy_2017_jamo_setting_best","")
tf.flags.DEFINE_string("lang", "kor","")
tf.flags.DEFINE_string("split", "root","")
tf.flags.DEFINE_integer("limitVocab", None,"")
tf.flags.DEFINE_string("trainData",'./data/kor/train.csv',"")
tf.flags.DEFINE_string("testData",'test.csv',"")
tf.flags.DEFINE_multi_integer("num_filters",[128],"")
tf.flags.DEFINE_multi_integer("hidden_unit",[32],"")

FLAGS = tf.flags.FLAGS


def asd(values):
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sess = tf.Session(config=config)
    set_session(sess)

    x_, y_, vocabulary, vocabulary_inv, max_word_length, labels = data_helper.load_data_kor(FLAGS.trainData, FLAGS.lang, FLAGS.split, FLAGS.limitVocab)
    x_ = np.array(list(x_))
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)
    x_train, _ = zip(*np.array(x_train))
    x_dev, _ = zip(*np.array(x_dev))
    x_train = np.array(x_train)
    x_dev = np.array(x_dev)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 300, input_shape=(None,)))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(values, activation=tf.nn.relu))
    model.add(keras.layers.Dense(6, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=tf.train.GradientDescentOptimizer(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    from keras.callbacks import EarlyStopping
    earlyStopping = EarlyStopping()

    history = model.fit(x_train,
                        y_train,
                        epochs=3000,
                        batch_size=128,
                        validation_data=(x_dev, y_dev),
                        verbose=1,
                        callbacks=[earlyStopping])
    history_dict = history.history

    test_file = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master2/CharCLSTMATTENT/data/kor/test10P.csv'

    x_, y_, max_word_length = data_helper.load_test_data_kor(test_file, 'root', vocabulary)
    x_ = np.array(list(x_))
    x_, raw_text = zip(*np.array(x_))
    loss_and_metrics = model.evaluate(np.array(x_),np.array(y_))
    print(loss_and_metrics)




    import matplotlib.pyplot as plt

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    with open('./mlp_test.txt','a') as f:
        f.write("param: {} | loss : {} , acc : {} , val_loss : {}, val_acc: {}\n".format(values,loss[len(loss)-1],acc[len(acc)-1],val_loss[len(val_loss)-1],val_acc[len(val_acc)-1]))
        f.write("Evaluate result: {}".format(loss_and_metrics))
        f.write("----------------------------------\n\n")
    #
    # epochs = range(1, len(acc) + 1)
    #
    # # "bo"는 "파란색 점"입니다
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b는 "파란 실선"입니다
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()


def mlp():
    x_, y_, vocabulary, vocabulary_inv, max_word_length, labels = data_helper.load_data_kor(FLAGS.trainData, FLAGS.lang, FLAGS.split, FLAGS.limitVocab)
    x_ = np.array(list(x_))
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)
    x_train, _ = zip(*np.array(x_train))
    x_dev, _ = zip(*np.array(x_dev))
    x_train = np.array(x_train)
    x_dev = np.array(x_dev)

    word_embeddings = data_helper.load_embeddings(vocabulary,300)
    embedding_mat = [word_embeddings[vocabulary_inv.get(idx)] for idx in vocabulary_inv]

    # Char-Embedding
    # char_embeddings = data_helper.load_char_embeddings(vocabulary)
    # embedding_mat = [char_embeddings[char] for index, char in enumerate(vocabulary_inv) if char in dict1]

    embedding_mat = np.array(embedding_mat, dtype = np.float32)

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    with tf.device('/gpu:0'), tf.name_scope('embedding'):
        W = tf.Variable(embedding_mat, name='W')

    embedded_chars = tf.nn.embedding_lookup(W, x_train)

    W1 = tf.Variable(tf.random_uniform([embedded_chars.shape[1].value,256,300],-1.,1.))
    W2 = tf.Variable(tf.random_uniform([256,6],-1.,1.))

    b1= tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([6]))

    L1 = tf.add(tf.matmul(embedded_chars,W1),b1)
    L1 = tf.nn.relu(L1)

    model = tf.add(tf.matmul(L1,W2),b2)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in range(100):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})

        if (step + 1) % 10 == 0:
            print(step + 1, sess.run(cost, feed_dict={X: x_train, Y: y_train}))

    #########
    # 결과 확인
    # 0: 기타 1: 포유류, 2: 조류
    ######
    # tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
    # 예) [[0 1 0] [1 0 0]] -> [1 0]
    #    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    predictions = sess.run(prediction, feed_dict={X: x_train})
    # print('예측값:', sess.run(prediction, feed_dict={X: x_train}))
    print('실제값:', sess.run(target, feed_dict={Y: y_train}))

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_train, Y: y_train}))


def free():
    x_, y_, vocabulary, vocabulary_inv, max_word_length, labels = data_helper.load_data_kor(FLAGS.trainData, FLAGS.lang, FLAGS.split, FLAGS.limitVocab)
    x_ = np.array(list(x_))
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)
    x_train, _ = zip(*np.array(x_train))
    x_dev, _ = zip(*np.array(x_dev))
    x_train = np.array(x_train)
    x_dev= np.array(x_dev)
    # x_train = np.array(x_train)
    # y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    # x_dev = tf.convert_to_tensor(x_dev, dtype=tf.int32)
    # y_dev = tf.convert_to_tensor(y_dev, dtype=tf.int32)
    # Parameters
    learning_rate = 0.1
    num_steps = 1000
    batch_size = 128
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = x_.shape[0]  # MNIST data input (img shape: 28*28)
    num_classes = 6  # MNIST total classes (0-9 digits)

    # Define the neural network
    def neural_net(x_dict):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, n_hidden_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, n_hidden_2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, num_classes)
        return out_layer

    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        # Build the neural network
        logits = neural_net(features)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs

    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_train, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=x_dev, y=y_dev,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'])


def train_cnn_rnn2(_params):
    tf.reset_default_graph()
    accuracy_list=[]
    start_vect = time.time()

    # x_, y_, vocabulary, vocabulary_inv,df, labels = data_helper.load_data_modified(input_file)
    # x_, y_, vocabulary, vocabulary_inv, labels,max_word_length = data_helper.load_data_modified_char(input_file)
    x_, y_, vocabulary, vocabulary_inv, max_word_length, labels = data_helper.load_data_kor(FLAGS.trainData,FLAGS.lang,FLAGS.split,FLAGS.limitVocab)

    # x_test, y_test = data_helper.load_test_data_kor(testData,FLAGS.split,vocabulary,vocabulary_inv)
    # x_test, raw_text = zip(*np.array(x_test))
    # x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)
    paramName, paramValue = list(zip(*_params[0]))
    params = {}
    for i, v in enumerate(paramName):
        params[v] = paramValue[i]
    params["hidden_unit"] = _params[1]
    params["num_filters"] = _params[2]


    print(params)
    # Assign a 300 dimension vector to each word
    word_embeddings = data_helper.load_embeddings(vocabulary,params['embedding_dim'])
    embedding_mat = [word_embeddings[vocabulary_inv.get(idx)] for idx in vocabulary_inv]

    # Char-Embedding
    # char_embeddings = data_helper.load_char_embeddings(vocabulary)
    # embedding_mat = [char_embeddings[char] for index, char in enumerate(vocabulary_inv) if char in dict1]

    embedding_mat = np.array(embedding_mat, dtype = np.float32)


    # Split the original dataset into train set and test set

    # x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)


    x_ = np.array(list(x_))
    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x_, y_, test_size=0.1)
    x_train, _ = zip(*np.array(x_train))
    x_dev, _ = zip(*np.array(x_dev))

    logging.info('x_train: {}, x_dev: {}'.format(len(x_train), len(x_dev)))
    logging.info('y_train: {}, y_dev: {}'.format(len(y_train), len(y_dev)))

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/media/gon/Volume", "runs", timestamp+"_"+str(params["hidden_unit"])+"_"+str(params["num_filters"])))


    cnn_rnn = TextCNNRNN(
        embedding_mat=embedding_mat,
        sequence_length=max_word_length,
        num_classes=y_train.shape[1],
        non_static=params['non_static'],
        hidden_unit=params['hidden_unit'],
        max_pool_size=params['max_pool_size'],
        filter_sizes=params['filter_sizes'].split(","),
        num_filters=params['num_filters'],
        embedding_size=params['embedding_dim'],
        l2_reg_lambda=params['l2_reg_lambda'])
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config=session_conf) as sess:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss=cnn_rnn.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(checkpoint_prefix)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        def real_len(batches):
            return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

        def train_step(x_batch, y_batch,writer = None):
            feed_dict = {
                cnn_rnn.input_x: x_batch,
                cnn_rnn.input_y: y_batch,
                cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
                cnn_rnn.batch_size: len(x_batch),
                cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                cnn_rnn.real_len: real_len(x_batch),
            }
            # print("{} : {}".format("pad.shape", np.shape(x_batch)))
            # print("{} : {}".format("asdwqffqw.shape", cnn_rnn.scores))
            # print("{} : {}".format("pad.shape", np.shape(y_batch)))
            # print("{} : {}".format("pad.shape", params['dropout_keep_prob']))
            # print("{} : {}".format("pad.shape", len(x_batch)))
            # print("{} : {}".format("pad.shape", np.shape(np.zeros([len(x_batch), 1, params['embedding_dim'], 1]))))
            # print("{} : {}".format("pad.shape", np.shape(real_len(x_batch))))
            _, step, loss, accuracy, summaries = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy, train_summary_op], feed_dict)

            if step % params['display_every'] == 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: {} step {}, loss {:g}, acc {:g}".format(time_str,epoch, step, loss, accuracy))
                writer.add_summary(summaries, step)

            return step

        def dev_step(x_batch, y_batch):
            feed_dict = {
                cnn_rnn.input_x: x_batch,
                cnn_rnn.input_y: y_batch,
                cnn_rnn.dropout_keep_prob: 1.0,
                cnn_rnn.batch_size: len(x_batch),
                cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                cnn_rnn.real_len: real_len(x_batch),
            }

            loss, accuracy, num_correct, predictions,summaries = sess.run(
                [cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions,dev_summary_op], feed_dict)

            return accuracy, loss, num_correct, predictions,summaries

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # Training starts here
        print("train_batches START")

        print("train_batches END")
        best_accuracy, best_at_step = 0, 0

        # Train the model with x_train and y_train

        def training(x_train, y_train,num_class):
            n_batch = int(math.ceil(len(x_train) / params["batch_size"]))
            m = 0
            for idx in range(n_batch):
                y_train_batch = np.zeros([params["batch_size"], num_class])
                x_train_batch = np.ndarray([params["batch_size"], max_word_length])
                for batch_num in range(params["batch_size"]):
                    y_train_batch[batch_num] = y_train[m]
                    x_train_batch[batch_num] = x_train[m]
                    m += 1
                    if m >= len(x_train):
                        m = 0
                step = train_step(x_train_batch, y_train_batch, writer=train_summary_writer)

            return step

        def validation(x_dev, y_dev,num_class):
            total_dev_correct = 0

            A_s = 0
            n_batch = int(math.ceil(len(x_dev) / params["batch_size"]))

            q = 0
            for idx in range(n_batch):
                y_dev_batch = np.zeros([params["batch_size"], num_class])
                x_dev_batch = np.ndarray([params["batch_size"], max_word_length])
                for batch_num in range(params["batch_size"]):
                    y_dev_batch[batch_num] = y_dev[q]
                    x_dev_batch[batch_num] = x_dev[q]
                    q += 1
                    if q >= len(x_dev):
                        q = 0

                accuracy, loss, num_dev_correct, predictions, summaries= dev_step(x_dev_batch, y_dev_batch)
                A_s += accuracy
                total_dev_correct += num_dev_correct
            dev_summary_writer.add_summary(summaries, current_step)
            return A_s/n_batch


        for epoch in range(params["num_epochs"]):
            current_step = tf.train.global_step(sess,global_step)

            training(x_train,y_train,y_train.shape[1])
            accuracy = validation(x_dev,y_dev,y_dev.shape[1])
            accuracy_list.append(accuracy)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print('Accuracy on dev set: {}'.format(accuracy))

            if accuracy > best_accuracy:
                best_accuracy, best_at_step = accuracy, current_step
                saver.save(sess, checkpoint_dir + "/best_model.ckpt")
                print('Saved model {} at step {}'.format(path, current_step))
                print('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            if current_step >= 500:
                if accuracy - np.mean(accuracy_list[int(round(len(accuracy_list) / 2)):]) <= 0.001:
                    print("Early Stopping")
                    break

                # Save the model files to trained_dir. predict.py needs trained model files.


            # Evaluate x_test and y_test
            # print('Training is complete, testing the best model on x_test and y_test')
            # print(checkpoint_prefix + '-' + str(best_at_step))
            # saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            # test_accuracy, prediction = test(x_test,y_test,y_test.shape[1])
            # print('Accuracy on test set: {}'.format(test_accuracy))

    # precision_recall_fscore_support(y_test,prediction)


    # Save trained parameters and files since predict.py needs them
    with open(checkpoint_dir + '/words_index.json', 'w') as outfile:
        json.dump((vocabulary,vocabulary_inv), outfile, indent=4, ensure_ascii=False)
    with open(checkpoint_dir + '/embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    with open(checkpoint_dir + '/labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)
    with open(checkpoint_dir + '/trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    duration = "%0.2f Min" % ((time.time() - start_vect) / 60)
    print("training Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))
    print("\n")
    del cnn_rnn
    return params,accuracy,checkpoint_dir, duration

# def train_cnn_rnn(input_file):
#     accuracy_list=[]
#     dict1 = list(r'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:"/\|_@#$%^&*~`+-=<>()[]{}') + ['\n', "'"]
#
#     start_vect = time.time()
#     # input_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/yahoo_answers_csv/train50000_2.csv'
#     input_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/ag_news_csv/train.csv'
#     # input_file = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master/data/train.csv.zip'
#     # input_file = sys.argv[1]
#
#     x_, y_, vocabulary, vocabulary_inv,df, labels = data_helper.load_data_modified(input_file)
#     # x_, y_, vocabulary, vocabulary_inv, labels,max_word_length = data_helper.load_data_modified_char(input_file)
#
#     # x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)
#
#
#     training_config = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master/training_config1.json'
#     # training_config = sys.argv[2]
#     params = json.loads(open(training_config).read())
#
#     # Assign a 300 dimension vector to each word
#     word_embeddings = data_helper.load_embeddings(vocabulary)
#     embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
#     # char_embeddings = data_helper.load_char_embeddings(vocabulary)
#     # embedding_mat = [char_embeddings[char] for index, char in enumerate(vocabulary_inv) if char in dict1]
#     embedding_mat = np.array(embedding_mat, dtype = np.float32)
#
#
#
#
#     # Split the original dataset into train set and test set
#     x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)
#
#     # Split the train set into train set and dev set
#     x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)
#
#     logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
#     logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
#
#     # Create a directory, everything related to the training will be saved in this directory
#     timestamp = str(int(time.time()))
#     out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#
#     # trained_dir = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master/result/trained_results_' + timestamp + '/'
#     # # if os.path.exists(trained_dir):
#     # #     shutil.rmtree(trained_dir)
#     # if not os.path.exists(trained_dir)
#     # os.makedirs(trained_dir)
#
#     graph = tf.Graph()
#     with graph.as_default():
#         session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             cnn_rnn = TextCNNRNN(
#                 embedding_mat=embedding_mat,
#                 sequence_length=x_train.shape[1],
#                 num_classes = y_train.shape[1],
#                 non_static=params['non_static'],
#                 hidden_unit=params['hidden_unit'],
#                 max_pool_size=params['max_pool_size'],
#                 filter_sizes=params['filter_sizes'],
#                 num_filters = params['num_filters'],
#                 embedding_size = params['embedding_dim'],
#                 l2_reg_lambda = params['l2_reg_lambda'])
#
#             global_step = tf.Variable(0, name='global_step', trainable=False)
#             optimizer = tf.train.AdamOptimizer(params['learning_rate'])
#             grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
#             train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#             # train_op = tf.train.AdamOptimizer(params['learning_rate'].minimize(cnn_rnn.loss,global_step=global_step))
#
#
#             # grad_summaries = []
#             # for g, v in grads_and_vars:
#             #     if g is not None:
#             #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
#             #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#             #         grad_summaries.append(grad_hist_summary)
#             #         grad_summaries.append(sparsity_summary)
#             #
#             # grad_summaries_merged = tf.summary.merge(grad_summaries)
#
#             # Checkpoint files will be saved in this directory during training
#             # checkpoint_dir = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master/result/checkpoints_' + timestamp + '/'
#             # if os.path.exists(checkpoint_dir):
#             #     shutil.rmtree(checkpoint_dir)
#             # os.makedirs(checkpoint_dir)
#             # checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
#
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#             checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             print(checkpoint_prefix)
#
#
#             # Summaries for loss and accuracy
#             loss_summary = tf.summary.scalar("loss", cnn_rnn.loss)
#             acc_summary = tf.summary.scalar("accuracy", cnn_rnn.accuracy)
#
#             # Train Summaries
#             train_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             train_summary_dir = os.path.join(out_dir, "summaries", "train")
#             train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
#             # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
#
#
#             def real_len(batches):
#                 return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]
#
#             def train_step(x_batch, y_batch,writer = None):
#                 feed_dict = {
#                     cnn_rnn.input_x: x_batch,
#                     cnn_rnn.input_y: y_batch,
#                     cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
#                     cnn_rnn.batch_size: len(x_batch),
#                     cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
#                     cnn_rnn.real_len: real_len(x_batch),
#                 }
#                 # print("{} : {}".format("pad.shape", np.shape(x_batch)))
#                 # print("{} : {}".format("pad.shape", np.shape(y_batch)))
#                 # print("{} : {}".format("pad.shape", params['dropout_keep_prob']))
#                 # print("{} : {}".format("pad.shape", len(x_batch)))
#                 # print("{} : {}".format("pad.shape", np.shape(np.zeros([len(x_batch), 1, params['embedding_dim'], 1]))))
#                 # print("{} : {}".format("pad.shape", np.shape(real_len(x_batch))))
#
#                 _, step, loss, accuracy, summaries = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy, train_summary_op,], feed_dict)
#
#
#                 if step % params['display_every'] == 0:
#                     time_str = datetime.datetime.now().isoformat()
#                     print("{}: {} step {}, loss {:g}, acc {:g}".format(time_str,epoch, step, loss, accuracy))
#                     writer.add_summary(summaries, step)
#
#
#
#
#             def dev_step(x_batch, y_batch):
#                 feed_dict = {
#                     cnn_rnn.input_x: x_batch,
#                     cnn_rnn.input_y: y_batch,
#                     cnn_rnn.dropout_keep_prob: 0.0,
#                     cnn_rnn.batch_size: len(x_batch),
#                     cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
#                     cnn_rnn.real_len: real_len(x_batch),
#                 }
#
#                 step, loss, accuracy, num_correct, predictions,summaries = sess.run(
#                     [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions,dev_summary_op], feed_dict)
#
#                 return accuracy, loss, num_correct, predictions,summaries
#
#             saver = tf.train.Saver()
#             sess.run(tf.global_variables_initializer())
#
#
#             # Training starts here
#             print("train_batches START")
#
#
#             print("train_batches END")
#             best_accuracy, best_at_step = 0, 0
#
#             # Train the model with x_train and y_train
#             for epoch in range(params["num_epochs"]):
#                 for train_batch in data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs']):
#
#                     x_train_batch, y_train_batch = zip(*train_batch)
#                     # x_train_batch = np.array(x_train_batch).transpose((1,0,2))
#
#
#                     # print("x_train_batch :",np.shape(x_train_batch))
#                     # print("y_train_batch :",np.shape(y_train_batch))
#                     # print("write feed_dict")
#                     # print("current_step :", current_step)
#
#                     # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#                     # Training log display
#                     train_step(x_train_batch,y_train_batch,writer=train_summary_writer)
#                     current_step = tf.train.global_step(sess,global_step)
#
#
#                     # # Evaluate the model with x_dev and y_dev
#                     # if current_step % params['evaluate_every'] == 0:
#
#                 total_dev_correct = 0
#
#
#                 for dev_batch in data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1):
#                     x_dev_batch, y_dev_batch = zip(*dev_batch)
#                     acc, loss, num_dev_correct, predictions,summaries = dev_step(x_dev_batch, y_dev_batch)
#                     total_dev_correct += num_dev_correct
#                 dev_summary_writer.add_summary(summaries, current_step)
#                 accuracy = float(total_dev_correct) / len(y_dev)
#                 accuracy_list.append(accuracy)
#                 print('Accuracy on dev set: {}'.format(accuracy))
#
#
#                 if accuracy >= best_accuracy:
#                     best_accuracy, best_at_step = accuracy, current_step
#                     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
#                     print('Saved model {} at step {}'.format(path, best_at_step))
#                     print('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
#
#
#                 # if current_step >= 500:
#                 #     if accuracy - np.mean(accuracy_list[int(round(len(accuracy_list) / 2)):]) <= 1.e5:
#                 #         print("early Stopping")
#                 #         break
#
#                 print('Training is complete, testing the best model on x_test and y_test')
#
#                 # Save the model files to trained_dir. predict.py needs trained model files.
#                 saver.save(sess, checkpoint_dir + "best_model.ckpt")
#
#                 # Evaluate x_test and y_test
#                 print(checkpoint_prefix + '-' + str(best_at_step))
#                 # print(list(sess))
#                 saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
#
#                 # test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
#                 # total_test_correct = 0
#                 # for test_batch in test_batches:
#                 #     x_test_batch, y_test_batch = zip(*test_batch)
#                 #     acc, loss, num_test_correct, predictions,_ = dev_step(x_test_batch, y_test_batch)
#                 #     total_test_correct += int(num_test_correct)
#                 # print('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
#
#     # Save trained parameters and files since predict.py needs them
#     with open(checkpoint_dir + '/words_index.json', 'w') as outfile:
#         json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
#     # with open(out_dir + 'embeddings.pickle', 'wb') as outfile:
#     #     pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
#     with open(checkpoint_dir + '/labels.json', 'w') as outfile:
#         json.dump(labels, outfile, indent=4, ensure_ascii=False)
#
#     params['sequence_length'] = x_train.shape[1]
#     with open(checkpoint_dir + '/trained_parameters.json', 'w') as outfile:
#         json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
#
#
#
#
#
#     print("training Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))
#
#     return best_accuracy

def getDatafilePath(root_path):
    for _, dirs,_ in os.walk(root_path):

        if dirs != []:
            print([root_path +dir + "/NNST_data.csv" for dir in dirs if dir != []])
            return [root_path +dir + "/NNST_data.csv" for dir in dirs if dir != []]




def process_make_dataframe(indexes):
    if os.path.exists('./acc/train/' + FLAGS.filename+".csv"):
        return DataFrame.from_csv("./acc/train/"+FLAGS.filename+".csv")
    else:
        return DataFrame(index=indexes)


def record_train_info(results):
    params, accuracy, checkpoint_dir, duration = results
    columns = [accuracy, checkpoint_dir.split("/")[-2], duration]
    indexes = ["Acc", "Directory", "Elapse Time"]
    df_csv = process_make_dataframe(indexes)
    df_csv[params["num_filters"]] = columns
    df_csv.to_csv('./acc/train/' + FLAGS.filename + ".csv")


def multiprocess_train_params(nump):
    # file = "/home/gon/Desktop/NNST-Naver-News-for-Standard-and-Technology-Database-master/nnst/NNST_data.csv"
    training_config = './training_config.json'
    params = json.loads(open(training_config).read())
    _params = [(v, params[v]) for v in params]
    _params = list(product([_params], FLAGS.hidden_unit,FLAGS.num_filters))
    results = train_cnn_rnn2(_params[0])
    # pool = Pool(nump)
    # results = list(tqdm.tqdm(pool.imap(train_cnn_rnn2,_params)))
    # pool.close()

    return results

if __name__ == '__main__':
    # python3 train.py ./data/train.csv.zip ./training_config.json
    # 0: Accuracy,

    # for i in [4,8,16,32,64,128,256,512,1024]:
    #     asd(i)

    results = multiprocess_train_params(1)
    record_train_info(results)
    # for result in results:
    #     record_train_info(result)



    # prediction, label = data_helper.process_make_report_no_rawtext(x_test, prediction, vocabulary_inv, y_test)
    # precision, recall, f1_score, _ = precision_recall_fscore_support(label, prediction, average=None)

    # indexes = ["Predict", "Label", "input_text", "raw_text"]
    # df_excel = DataFrame()
    # df_excel["Predict"] = prediction
    # df_excel["Label"] = label
    # df_excel["input_text"] = text
    # df_excel["raw_text"] = raw_text
    # # input datas in Dataframe
    # df_excel.to_excel(out_dir+"/"+FLAGS.filename+".xlsx")

