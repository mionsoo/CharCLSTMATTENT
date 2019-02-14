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
from cnn import TextCNNRNN
from sklearn.model_selection import train_test_split


logging.getLogger().setLevel(logging.INFO)

tf.flags.DEFINE_string("filename", "Accuracy_C128","")
tf.flags.DEFINE_string("lang", "kor","")
tf.flags.DEFINE_string("split", "root","")
tf.flags.DEFINE_integer("limitVocab", None,"")

FLAGS = tf.flags.FLAGS

def train_cnn_rnn2(input_file):
    accuracy_list=[]
    start_vect = time.time()

    # x_, y_, vocabulary, vocabulary_inv,df, labels = data_helper.load_data_modified(input_file)
    # x_, y_, vocabulary, vocabulary_inv, labels,max_word_length = data_helper.load_data_modified_char(input_file)
    x_, y_, vocabulary, vocabulary_inv, max_word_length, labels = data_helper.load_data_kor(input_file,FLAGS.lang,FLAGS.split,FLAGS.limitVocab)

    # x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)

    training_config = './training_config.json'
    params = json.loads(open(training_config).read())

    # Assign a 300 dimension vector to each word
    word_embeddings = data_helper.load_embeddings(vocabulary,params['embedding_dim'])
    embedding_mat = [word_embeddings[vocabulary_inv.get(idx)] for idx in vocabulary_inv]

    # Char-Embedding
    # char_embeddings = data_helper.load_char_embeddings(vocabulary)
    # embedding_mat = [char_embeddings[char] for index, char in enumerate(vocabulary_inv) if char in dict1]

    embedding_mat = np.array(embedding_mat, dtype = np.float32)


    # Split the original dataset into train set and test set
    x_ = np.array(list(x_))
    x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)
    x_test, raw_text = zip(*np.array(x_test))
    x,_ = zip(*np.array(x))
    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/media/gon/Volume", "runs", timestamp))

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        sess = tf.InteractiveSession(config=session_conf)
        with sess.as_default():
            cnn_rnn = TextCNNRNN(
                embedding_mat=embedding_mat,
                sequence_length=max_word_length,
                num_classes = y_train.shape[1],
                non_static=params['non_static'],
                hidden_unit=params['hidden_unit'],
                max_pool_size=params['max_pool_size'],
                filter_sizes=params['filter_sizes'].split(","),
                num_filters = params['num_filters'],
                embedding_size = params['embedding_dim'],
                l2_reg_lambda = params['l2_reg_lambda'])

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

            def test(x_test, y_test,num_class):
                total_test_correct = 0
                predictionList = []

                A_s = 0
                n_batch = int(math.ceil(len(x_test) / params["batch_size"]))

                q = 0
                for idx in range(n_batch):
                    y_test_batch = np.zeros([params["batch_size"], num_class])
                    x_test_batch = np.ndarray([params["batch_size"], max_word_length])
                    for batch_num in range(params["batch_size"]):
                        y_test_batch[batch_num] = y_test[q]
                        x_test_batch[batch_num] = x_test[q]
                        q += 1
                        if q >= len(x_test):
                            q = 0

                    accuracy, loss, num_test_correct, predictions, summaries= dev_step(x_test_batch, y_test_batch)
                    A_s += accuracy
                    total_test_correct += num_test_correct
                    predictionList.append(predictions)

                return A_s/n_batch, predictionList


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
            print('Training is complete, testing the best model on x_test and y_test')
            print(checkpoint_prefix + '-' + str(best_at_step))
            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_accuracy, prediction = test(x_test,y_test,y_test.shape[1])
            print('Accuracy on test set: {}'.format(test_accuracy))



    # Save trained parameters and files since predict.py needs them
    with open(checkpoint_dir + '/words_index.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    with open(checkpoint_dir + '/embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    with open(checkpoint_dir + '/labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)
    with open(checkpoint_dir + '/trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    duration = "%0.2f Min" % ((time.time() - start_vect) / 60)
    print("training Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))
    print("\n")

    return test_accuracy, accuracy, checkpoint_dir, duration, raw_text, prediction ,x_test,y_test, vocabulary_inv ,out_dir

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
    if os.path.exists('./acc/' + FLAGS.filename):
        return DataFrame.from_csv(root_path + FLAGS.filename)
    else:
        return DataFrame(index=indexes)




if __name__ == '__main__':
    # python3 train.py ./data/train.csv.zip ./training_config.json
    # 0: Accuracy,
    data_accuracy = {}
    # file = "/home/gon/Desktop/NNST-Naver-News-for-Standard-and-Technology-Database-master/nnst/NNST_data.csv"
    root_path = './data/'
    path = getDatafilePath(root_path)[0]
    print(path)
    print("Dataset : ",path.split("/")[3])

    test_accuracy, accuracy, checkpoint_dir, duration,raw_text,prediction,x_test,y_test,vocabulary_inv,out_dir = train_cnn_rnn2(path)

    prediction, label, text, raw_text = data_helper.process_make_report(x_test,prediction,raw_text,vocabulary_inv,y_test)


    columns = [accuracy, test_accuracy, precision,recall,f1_score,checkpoint_dir.split("/")[-2], duration]
    indexes = ["Acc", "Test_Acc", "Precision","Recall","F1_Score","Directory", "Elapse Time"]
    df_csv = process_make_dataframe(indexes)
    df_csv[path.split("/")[3]] = columns
    df_csv.to_csv('./acc/' + FLAGS.filename + ".csv")


    indexes = ["Predict", "Label", "input_text", "raw_text"]
    df_excel = DataFrame()
    df_excel["Predict"] = prediction
    df_excel["Label"] = label
    df_excel["input_text"] = text
    df_excel["raw_text"] = raw_text
    # input datas in Dataframe
    df_excel.to_excel(out_dir+"/"+FLAGS.filename+".xlsx")

