import os
import time
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import tqdm
from hangul_utils import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool

# CNN
from cnn import TextCNNRNN

# RNN
# from rnn import TextCNNRNN

# CRNN
# from char_cnn_biLstm2 import TextCNNRNN

logging.getLogger().setLevel(logging.INFO)

timestamps = '1555905877_32_128'.split('\t')
tf.flags.DEFINE_string("filename",'cnn_root_Accuracy_2017_jamo_setting_best_10000',"")
tf.flags.DEFINE_multi_string("timestamp", timestamps, "")

FLAGS = tf.flags.FLAGS


def load_trained_params(trained_dir):
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())

    with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
        fetched_embedding = pickle.load(input_file)
    embedding_mat = np.array(fetched_embedding, dtype = np.float32)
    return params, words_index, labels, embedding_mat


def load_trained_params_modified(trained_dir):
    params = json.loads(open(trained_dir + 'trained_parameters.json').read())
    words_index = json.loads(open(trained_dir + 'words_index.json').read())
    labels = json.loads(open(trained_dir + 'labels.json').read())

    with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
        fetched_embedding = pickle.load(input_file)
    embedding_mat = np.array(fetched_embedding, dtype = np.float32)

    return params, words_index, labels, embedding_mat


def stopwordElimination(sentence):
    stop_words = set(stopwords.words('english'))
    word_token = word_tokenize(sentence)
    filtered_sentence = [word for word in word_token if word not in stop_words if word not in [',','"','\\',]]
    # filtered_fullsentence = filtered_sentence[0]+filtered_sentence[1:]
    filtered_fullsentence = filtered_sentence[0] + "|" + " ".join(filtered_sentence[1:])
    # print(filtered_fullsentence)
    return filtered_fullsentence


def load_test_data(test_file,labels):
    df = pd.read_csv(test_file, sep='|')
    select = ['Descript']

    df = df.dropna(axis=0, how='any', subset=select)
    test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_ = None
    if 'Category' in df.columns:
        select.append('Category')
        y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

    not_select = list(set(df.columns) - set(select))
    df = df.drop(not_select, axis=1)
    return test_examples, y_, df


def load_test_data2(test_file,labels):
    df = pd.read_csv(test_file)
    select = ['total']
    df = df.fillna('')
    print(len(df.columns))
    # df2['total'] =  df2['title'] + ' ' + df2['context']+ ' ' + df2['answer'].map(str)
    if len(df.columns) == 3:
        df.columns = ['label', 'title', 'context']
        df['total'] = df[['context', 'title']].apply(lambda x: ' '.join(x), axis=1)
    elif len(df.columns) == 4:
        df.columns = ['label', 'title', 'context', 'answer']
        df['total'] = df[['answer', 'context', 'title']].apply(lambda x: ' '.join(x), axis=1)
    else:
        df.columns = ['label', 'title']
        df['total'] = df[['title']].apply(lambda x: ' '.join(x), axis=1)
    df = df.dropna(axis=0, how='any', subset=select)
    test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    y_ = None
    if 'label' in df.columns:
        select.append('label')
        y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

    df = pd.read_csv(test_file)
    # not_select = list(set(df.columns) - set(select))
    # df = df.drop(not_select, axis=1)
    return test_examples, y_


def load_test_data3(test_file,labels):
    list_data = []
    labels = []
    total = ""
    actual_max_length = 1014
    with open(test_file, 'r', encoding="utf8") as tsv:
        for line in tsv:
            lineset = stopwordElimination(line)
            sep = lineset.split("|")
            sentence_class = int(sep[0].replace("\ufeff", ""))
            sentence_english = sep[1].lower()
            labels.append(sentence_class)
            # for i in word_tokenize(sentence_english):
            #     # print(i)
            #     actual_max_word_length = max(actual_max_length,len(i))
            total += sentence_english
            list_data.append([sentence_english, sentence_class])

    dic = list(set(total))  # id -> char
    # dic.insert(0, "P")  # P:PAD symbol(0)
    dict1 = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n")
    vocabulary = {w: i for i, w in enumerate(dict1)}
    CLASS_SIZE = len(set([c[1] for c in list_data]))
    test_examples = [[[vocabulary[char] for char in data[0] if char in dict1], data[1]] for data in list_data]

    data_x = np.zeros((len(test_examples), actual_max_length), dtype=np.int)
    data_y = np.zeros((len(test_examples), CLASS_SIZE), dtype=np.int)

    # index = np.random.choice(range(len(list_data)), size, replace=False)
    # index = len(list_data)
    for idx in range(len(test_examples)):
        x = test_examples[idx][0]
        x = x[:actual_max_length - 1] + [0] * max(actual_max_length - len(x), 1)
        y = test_examples[idx][1]
        y = np.eye(CLASS_SIZE)[y-1]

        data_x[idx] = np.array(x)
        data_y[idx] = y


    # not_select = list(set(df.columns) - set(select))
    # df = df.drop(not_select, axis=1)
    return data_x, data_y,actual_max_length


def map_word_to_index(examples, words_index):
    x_ = []
    for example in examples:
        temp = []
        for word in example:
            if word in words_index:
                temp.append(words_index[word])
            else:
                temp.append(0)
        x_.append(temp)
    return x_


def load_test_data_kor(file,split,vocabulary):
    df = pd.read_csv(file)
    labels2 = sorted(list(set(df["class"].tolist())))
    num_labels = len(labels2)
    one_hot2 = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot2, 1)
    label_dict = dict(zip(labels2, one_hot2))
    rawText = df["text"]
    df["raw_text"] = rawText

    x_raw = data_helper.preprocess_data_by_split_type(df, split)
    y_raw = df["class"].apply(lambda y: label_dict[y]).tolist()
    max_length = max([len(i) for i in x_raw])

    x_raw = data_helper.padding(x_raw, vocabulary, max_length)

    x = zip(np.array(x_raw), rawText)
    y = np.array(y_raw)

    return x, y, max_length


def recalculate(args):
    args = np.concatenate(np.array(args))
    args = [float("%0.4f" % (i)) for i in args]
    args = np.array(args).reshape(3,6)

    return args


def process_make_dataframe(indexes,extension):
    if os.path.exists('./acc/test/' + FLAGS.filename+"."+extension):
        return pd.DataFrame.from_csv("./acc/test/"+FLAGS.filename+"."+extension)
    else:
        return pd.DataFrame(index=indexes)


def predict_unseen_data(timestamp):
    tf.reset_default_graph()
    start_vect = time.time()
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/ag_news_csv/test.csv'
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/sogou_news_csv/test.csv'
    test_file = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master2/CharCLSTMATTENT/data/kor/test.csv'
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/yelp_review_full_csv/test.csv'
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/yelp_review_polarity_csv/test.csv'
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/dbpedia_csv/test.csv'
    # test_file = '/home/gon/Desktop/rnn-text-classification-tf-master/data/yahoo_answers_csv/test500.csv'
    # test_file = '/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master/data/small_samples.csv'
    trained_dir ='/media/gon/Volume/runs/'+timestamp+'/checkpoints/'
    # trained_dir = sys.argv[1]
    if not trained_dir.endswith('/'):
        trained_dir += '/'
    # test_file = sys.argv[2]

    params, words_index, labels, embedding_mat = load_trained_params_modified(trained_dir)
    vocabulary, vocabulary_inv = words_index
    x_, y_, max_word_length = load_test_data_kor(test_file, 'root',vocabulary)
    x_ = np.array(list(x_))
    x_, raw_text = zip(*np.array(x_))
    # x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
    # x_ = map_word_to_index(x_, words_index)

    # x_test, y_test = np.asarray(x_), None
    # if y_ is not None:
    #     y_test = np.asarray(y_)

    timestamp = trained_dir.split('/')[-2].split('_')[-1]
    predicted_dir = './predicted_results_' + timestamp + '/'

    cnn_rnn = TextCNNRNN(
        embedding_mat=embedding_mat,
        non_static=params['non_static'],
        hidden_unit=params['hidden_unit'],
        sequence_length=max_word_length,
        max_pool_size=params['max_pool_size'],
        filter_sizes=params['filter_sizes'].split(','),
        num_filters=params['num_filters'],
        num_classes=len(labels),
        embedding_size=params['embedding_dim'],
        l2_reg_lambda=params['l2_reg_lambda'],
        tficf=FLAGS.tficf)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True

    with tf.Session(config=session_conf) as sess:
        # sess.run(tf.global_variables_initializer())
        def real_len(batches):
            return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

        # def predict_step(x_batch):
        #     feed_dict = {
        #         cnn_rnn.input_x: x_batch,
        #         cnn_rnn.input_y: y_batch,
        #         cnn_rnn.dropout_keep_prob: 1.0,
        #         cnn_rnn.batch_size: len(x_batch),
        #         cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
        #         cnn_rnn.real_len: real_len(x_batch),
        #     }
        #     predictions = sess.run([cnn_rnn.predictions], feed_dict)
        #     return predictions

        checkpoint_file = trained_dir + 'best_model.ckpt'


        saver = tf.train.Saver(tf.global_variables())
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        logging.critical('{} has been loaded'.format(checkpoint_file))

        # batches = data_helper.batch_iter(list(x_), params['batch_size'], 1, shuffle=False)

        predictions, predict_labels = [], []
        A_s=0
        def dev_step(x_batch, y_batch):
            feed_dict = {
                cnn_rnn.input_x: x_batch,
                cnn_rnn.input_y: y_batch,
                cnn_rnn.dropout_keep_prob: 1.0,
                cnn_rnn.batch_size: len(x_batch),
                cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
                cnn_rnn.real_len: real_len(x_batch),
            }

            loss, accuracy, num_correct, predictions = sess.run(
                [cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)

            # print("input : ",inputX)

            return accuracy, loss, num_correct, predictions
        def validation(x_dev, y_dev,num_class):
            total_dev_correct = 0
            predictionList = []

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
                        print(q)
                        q = 0

                accuracy, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
                A_s += accuracy
                total_dev_correct += num_dev_correct
                predictionList.append(predictions)


            # accuracy = float(total_dev_correct) / len(y_dev)
            # accuracy_list.append(accuracy)


            return A_s/n_batch, predictionList,loss

        accuracy,predictions,loss = validation(x_,y_,y_.shape[1])
        # for x_batch in batches:
        #     batch_predictions = predict_step(x_batch)[0]
        #
        #     for batch_prediction in batch_predictions:
        #
        #         predictions.append(batch_prediction)
        #         predict_labels.append(labels[batch_prediction])

        # Save the predictions back to file
        # df['NEW_PREDICTED'] = predict_labels
        # columns = sorted(df.columns, reverse=True)
        # print(df)
        # df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=['label','NEW_PREDICTED','title','context','answer'])

        if y_ is not None:
            # y_test = np.array(np.argmax(y_, axis=1))
            # print(np.array(predictions))
            # accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
            print('The accuracy is: {}'.format(accuracy))


        logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))
    del cnn_rnn
    return accuracy, params, predictions, raw_text, start_vect, trained_dir, vocabulary_inv, x_, y_,predicted_dir,loss


def record_test_output(result):
    accuracy, params, predictions, raw_text, start_vect, trained_dir, vocabulary_inv, x_, y_,predicted_dir, loss = result
    prediction, label, text, raw_text = data_helper.process_make_report(x_, predictions, raw_text, vocabulary_inv, y_)
    precision, recall, f1_score, _ = precision_recall_fscore_support(label, prediction, average=None)
    precision, recall, f1_score = recalculate((precision, recall, f1_score))

    print(confusion_matrix(label,prediction))

    duration = "%0.2f Min" % ((time.time() - start_vect) / 60)
    columns = [accuracy, loss, precision, recall, f1_score, trained_dir.split("/")[-3], duration]
    indexes = ["Acc", "Loss", "Precision", "Recall", "F1_Score", "Directory", "Elapse Time"]
    df_csv = process_make_dataframe(indexes, "csv")
    df_csv[params["num_filters"]] = columns
    df_csv.to_csv('./acc/test/' + FLAGS.filename + ".csv")

    # indexes = ["Predict", "Label", "input_text", "raw_text"]
    df_excel = process_make_dataframe(None,"xlsx")
    df_excel["Predict"] = prediction
    df_excel["Label"] = label
    df_excel["input_text"] = text
    df_excel["raw_text"] = raw_text
    # input datas in Dataframe
    df_excel.to_excel(trained_dir+"/"+FLAGS.filename+".xlsx")
    df_excel.to_csv(trained_dir + "/" + FLAGS.filename + ".csv")



if __name__ == '__main__':
    # python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv

    for i in FLAGS.timestamp:
        result = predict_unseen_data(i)
        record_test_output(result)

    # results = multiprocessing_param_test(3)
    # for result in results:
    #     record_test_output(result)
    # python3 predict.py ./trained_results_1538469237/ ./data/small_samples.csv