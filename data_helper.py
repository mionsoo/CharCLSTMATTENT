import re
import tqdm
import logging
import itertools
import numpy as np
import pandas as pd
import random
from hangul_utils import word_tokenize
from nltk.corpus import stopwords

from collections import Counter


logging.getLogger().setLevel(logging.INFO)


def stopwordElimination(sentence):
    stop_words = set(stopwords.words('english'))
    word_token = word_tokenize(sentence)
    filtered_sentence = [word for word in word_token if word not in stop_words if word not in [',','"','\\',]]
    # filtered_fullsentence = filtered_sentence[0]+filtered_sentence[1:]
    filtered_fullsentence = filtered_sentence[0] + "|" + " ".join(filtered_sentence[1:])
    # print(filtered_fullsentence)
    return filtered_fullsentence



def clean_str(s):
    # s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    # s = re.sub(r" : ", ":", s)
    # s = re.sub(r"\'s", " \'s", s)
    # s = re.sub(r"\'ve", " \'ve", s)
    # s = re.sub(r"n\'t", " n\'t", s)
    # s = re.sub(r"\'re", " \'re", s)
    # s = re.sub(r"\'d", " \'d", s)
    # s = re.sub(r"\'ll", " \'ll", s)
    # s = re.sub(r",", " , ", s)
    # s = re.sub(r"!", " ! ", s)
    s = re.sub(r'"', '\"', s)
    # s = re.sub(r"\(", " \( ", s)
    # s = re.sub(r"\)", " \) ", s)
    # s = re.sub(r"\?", " \? ", s)
    # s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()


def load_embeddings(vocabulary):
    word_embeddings = {}
    # dic = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n ")

    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings

def load_char_embeddings(vocabulary):
    char_embeddings = {}
    dict1 = list(r'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:"/\|_@#$%^&*~`+-=<>()[]{}') + ['\n', "'"]
    # print(vocabulary,len(dict))
    for i,char in enumerate(vocabulary):
        c = char
        if c in dict1:
            char_embeddings[char] = np.random.uniform(-0.25, 0.25, len(dict1))
    return char_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else: # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    # for epoch in range(num_epochs):
        # print("epoch : ",epoch)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        # print("batch_num : ",batch_num)
        yield shuffled_data[start_index:end_index]


def load_data(filename):
    df = pd.read_csv(filename, compression='zip')
    selected = ['Category', 'Descript']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))

    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    x_raw = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)
    return x, y, vocabulary, vocabulary_inv, df, labels


def load_data_modified(filename):
    print(filename)
    df2 = pd.read_csv(filename)
    selected2 = ['label','total']
    # selected3 = ['label','title','context','answer']
    df2 = df2.fillna('')
    print(len(df2.columns))
    # df2['total'] =  df2['title'] + ' ' + df2['context']+ ' ' + df2['answer'].map(str)
    if len(df2.columns) == 3:
        df2.columns = ['label','title','context']
        df2['total'] = df2[['context', 'title']].apply(lambda x: ' '.join(x), axis=1)
    elif len(df2.columns) == 4:
        df2.columns = ['label','title','context','answer']
        df2['total'] = df2[['answer', 'context', 'title']].apply(lambda x: ' '.join(x), axis=1)
    else:
        df2.columns = ['label','title']
        df2['total'] = df2[['title']].apply(lambda x: ' '.join(x), axis=1)
    '''
    for idx in tqdm.tqdm(range(len(df2))):
        if type(df2['title'][idx]) is float:
            df2['title'][idx] = ''
        if type(df2['context'][idx]) is float:
            df2['context'][idx] = ''
        if type(df2['answer'][idx]) is float:
            df2['answer'][idx] = ''
        # print(type(df2['title'][idx]),df2['title'][idx])
        # print(type(df2['context'][idx]),df2['context'][idx])
        # print(type(df2['answer'][idx]),df2['answer'][idx],'\n')
        df2['title'][idx] = df2['title'][idx] + ' ' + df2['context'][idx]+ ' ' + df2['answer'][idx]
    '''
    # df2[selected2[1]] = df2[selected3[1]]+' '+df2[selected3[2]]+' '+df2[selected3[3]]
    non_selected2 = list(set(df2.columns) - set(selected2))
    df2 = df2.drop(non_selected2, axis=1)
    # df2 = df2.dropna(axis=0, how='any', subset=selected2)
    df2 = df2.reindex(np.random.permutation(df2.index))

    labels2 = sorted(list(set(df2[selected2[0]].tolist())))
    num_labels = len(labels2)
    one_hot2 = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot2, 1)
    label_dict = dict(zip(labels2, one_hot2))
    x_raw = df2[selected2[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
    y_raw = df2[selected2[0]].apply(lambda y: label_dict[y]).tolist()

    print("x_raw.shape",np.shape(x_raw))
    x_raw = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(x_raw)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)
    return x, y, vocabulary, vocabulary_inv, df2, labels2


def load_data_modified_char(filename):

    list_data = []
    labels = []
    total = ""
    actual_max_length = 90
    with open(filename, 'r', encoding="utf8") as tsv:
        for line in tsv:
            lineset = stopwordElimination(line)
            sep = lineset.split("|")
            sentence_class = int(sep[0].replace("\ufeff", ""))
            sentence_english = sep[1].lower()
            labels.append(sentence_class)
            # for i in word_tokenize(sentence_english):
            #     # print(i)
            #     actual_max_word_length = max(actual_max_word_length,len(i))
            total += sentence_english
            list_data.append([sentence_english, sentence_class])
    # print(sentence_english,clean_str(sentence_english))
    # dict1 = list(r"abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}") + ['\n']
    dict1 = list(r'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:"/\|_@#$%^&*~`+-=<>()[]{}') + ['\n', "'"]

    labels = list(set(labels))
    dic = list(set(total))  # id -> char
    # dic.insert(0, "P")  # P:PAD symbol(0)
    vocabulary = {w: i for i, w in enumerate(dict1)}
    vocabulary_inv = [w for w in dict1]

    # print("RDIC")
    # print(vocabulary)
    #
    # VOCAB_SIZE = len(dic)
    # print("VOCABULARY SIZE : %d" % VOCAB_SIZE)

    CLASS_SIZE = len(set([c[1] for c in list_data]))
    print("CLASS SIZE : %d" % CLASS_SIZE)
    # print(list_data[-1][0])

    list_data.sort(key=lambda s: len(s[0]))
    MAX_LEN = len(list_data[-1][0]) + 1
    print("SENTENCE MAX LEN : %d" % MAX_LEN)
    print("Word Max Len : %d" % actual_max_length)
    # [[[print(char) for char in data[0]], data[1]] for data in list_data]

    list_data = [[[vocabulary[char] for char in data[0] if char in dict1], data[1]] for data in list_data]



    # random.shuffle(list_data, random.random)
    #
    # list_data_test = list_data[:200]
    # list_data = list_data[200:]
    # print("TOTAL TRAIN DATASET : %d" % (len(list_data)))
    # print("TOTAL TEST DATASET : %d" % (len(list_data_test)))
    # assert size <= len(list_data)

    data_x = np.zeros((len(list_data), actual_max_length), dtype=np.int)
    data_y = np.zeros((len(list_data), CLASS_SIZE), dtype=np.int)

    # index = np.random.choice(range(len(list_data)), size, replace=False)
    # index = len(list_data)
    for idx in reversed(range(len(list_data))):
        x = list_data[idx][0]
        x = x[:actual_max_length - 1] + [0] * max(actual_max_length - len(x), 1)
        y = list_data[idx][1]
        y = np.eye(CLASS_SIZE)[y-1]

        data_x[idx] = np.array(x)
        data_y[idx] = y

    # vocabulary = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

    return data_x,data_y,vocabulary,vocabulary_inv,labels,actual_max_length

if __name__ == "__main__":
    train_file = './data/train.csv.zip'
    load_data(train_file)
