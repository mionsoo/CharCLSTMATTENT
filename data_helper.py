import re
import os
import pickle
import tqdm
import logging
import itertools
import numpy as np
import pandas as pd
import hanja
import time
from flashtext.keyword import KeywordProcessor
from KoreanTagger import MeCabParser
from hangul_utils import word_tokenize
from nltk.corpus import stopwords
from math import log

from collections import Counter
from multiprocessing import Pool

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


def load_category_embeddings(vocabulary):
    word_embeddings = {}
    # dic = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n ")

    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 384)
    return word_embeddings


def load_embeddings(vocabulary,embeddingDim):
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


def categoryEmbedding(categoryClusters,num_filters,filter_sizes):
    categoryEmbeddingMats=[]
    for categoryCluster in categoryClusters:
        _vocabulary,_vocabulary_inv = build_category_vocab(categoryCluster)
        word_embeddings = load_category_embeddings(_vocabulary)
        embedding_mat = [word_embeddings[_vocabulary_inv.get(idx)] for idx in _vocabulary_inv]
        categoryEmbeddingMats.append(embedding_mat)
    return np.concatenate(categoryEmbeddingMats,0)


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
            print(sentence)
            print(type(sentence))
            padded_sentence = sentence + ([padding_word] * num_padding)
        padded_sentences.append(padded_sentence)
    return padded_sentences, sequence_length


def build_vocab(sentences, limitVocab):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = {idx: word[0] for idx, word in enumerate(word_counts.most_common(limitVocab))}
    vocabulary = {vocabulary_inv.get(i) : i for i in vocabulary_inv}
    return vocabulary, vocabulary_inv

def build_category_vocab(sentences):

    vocabulary_inv = {idx: word for idx, word in enumerate(sentences)}
    vocabulary = {vocabulary_inv.get(i) : i for i in vocabulary_inv}
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


def load_test_data_kor(file,split,vocabulary):
    df = pd.read_csv(file)
    labels2 = sorted(list(set(df["class"].tolist())))
    num_labels = len(labels2)
    one_hot2 = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot2, 1)
    label_dict = dict(zip(labels2, one_hot2))
    rawText = df["text"]
    df["raw_text"] = rawText

    x_raw = preprocess_data_by_split_type(df, split)
    y_raw = df["class"].apply(lambda y: label_dict[y]).tolist()
    max_length = max([len(i) for i in x_raw])

    x_raw = padding(x_raw, vocabulary, max_length)
    # df.sort_values()
    x = zip(np.array(x_raw), rawText)
    y = np.array(y_raw)

    return x, y, max_length

def makeCategoryCluster(a):
    # a = {i for i in a if i != {}}
    culture = []
    digital = []
    economic = []
    foreign = []
    politics = []
    society = []
    for i in a:
        # print("asdasd",i)
        for word in wordList:
            value= i.get(word)
            # print("vqwqwv",word)
            # print(i.get(word))
            try:
                value.index(1)
            except:
                pass
            else:
                if value == None:
                    pass
                elif value.index(1) +1 == 1:
                    culture.append(word)
                elif value.index(1) +1 == 2:
                    digital.append(word)
                elif value.index(1) +1 == 3:
                    economic.append(word)
                elif value.index(1) +1 == 4:
                    foreign.append(word)
                elif value.index(1) +1 == 5:
                    politics.append(word)
                elif value.index(1) +1 == 6:
                    society.append(word)
                else:
                    pass
    return np.array([culture, digital, economic, foreign, politics, society])


def get_label(df, labelq):
    labelq[df-1] = 1
    return labelq

def load_data_kor(file, lang, split, limitVocab, tficf):
    # file = "/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master2/CharCLSTMATTENT/data/kor/train1P.csv"
    pklFile = "/home/gon/Desktop/multi-class-text-classification-cnn-rnn-master2/CharCLSTMATTENT/pkl/"+ lang + "/"+ split + "/"+ file.split("/")[-1].split(".")[0] + "_data_preprocess.pkl"
    if not check_pickle_file(pklFile):
        with open(pklFile,"wb") as f:
            df = pd.read_csv(file)
            labels2 = sorted(list(set(df["class"].tolist())))
            num_labels = len(labels2)
            one_hot2 = np.zeros((num_labels, num_labels), int)
            np.fill_diagonal(one_hot2, 1)
            label_dict = dict(zip(labels2, one_hot2))
            rawText = df["text"]
            df["raw_text"] = rawText

            x_raw = preprocess_data_by_split_type(df, split)
            y_raw = df["class"].apply(lambda y: label_dict[y]).tolist()
            max_length = max([len(i)for i in x_raw])
            pickle.dump((x_raw, y_raw, labels2, rawText,max_length,df),f)
    else:
        with open(pklFile,"rb") as f:
            x_raw, y_raw, labels2, rawText, max_length,df = pickle.load(f)

    vocabulary, vocabulary_inv = build_vocab(x_raw, limitVocab)
    vocabulary["<PAD>"] = len(vocabulary)
    vocabulary_inv[len(vocabulary_inv)] = "<PAD>"
    vocabulary["<UNK>"] = len(vocabulary)
    vocabulary_inv[len(vocabulary_inv)] = "<UNK>"

    categoryEmbeddingMats = 0
    if tficf:
        print("make category clustering")
        # Extract Category Clustering using Tficf
        global wordList
        # wordList = Counter(list(itertools.chain(*x_raw)))
        wordList = vocabulary_inv.values()

        with open("./categoryData.pkl",'wb') as f:
            df2 = df
            df2["text"] = x_raw
            df2["class"] = [list(i).index(1)+1 for i in np.array(y_raw)]
            pickle.dump(df2,f)

        # pool = Pool(nump)
        # results = list(tqdm.tqdm(pool.imap(train_cnn_rnn2,_params)))
        # pool.close()

        q ={}
        z = []
        pool = Pool(20)

        a = list(range(0,len(df2.values),100))
        fr = a[:-1]
        to = a[1:]



        frto = list(zip(fr, to))

        s = list(tqdm.tqdm(pool.imap(madeCategoryClusters,frto)))
        for i in s:
            z.append(i)


        categoryClusters = makeCategoryCluster(z)
        categoryEmbeddingMats = categoryEmbedding(categoryClusters, 128, "6,9,12")

    x_raw = padding(x_raw,vocabulary,max_length)
    # x_raw, max_length = pad_sentences(x_raw)


    x = zip(np.array(x_raw), rawText)
    y = np.array(y_raw)


    return x, y, vocabulary, vocabulary_inv, max_length, labels2,categoryEmbeddingMats






def madeCategoryClusters(frto):
    fr,to = frto
    start_vect = time.time()
    a = {}
    with open("./categoryData.pkl",'rb')as f:
        df2 = pickle.load(f)



    result = {}
    # cf = calculate_cf(word, df)

    labelq = [0] * 6
    for _, la, line, row, in df2.values[fr:to]:
        for word in wordList:
            if word in line:
                if result.__contains__(word):
                    result[word][la-1] += 1
                else:
                    result[word] = [0 for _ in range(6)]
                    result[word][la-1] += 1

                tficf = Counter(wordList).get(word) * np.log2((6 / 1 + np.sum(result[word])))
                if tficf >= log(6) + 1 and np.sum(result[word]) == 1:
                    a[word] = result[word]
            else:
                pass


    # print("training Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))
    return a


def padding(r_data,vocab,max_length):
    data = []
    for raw in r_data:
        d = []
        for w in raw:
            idx = vocab.get(w)
            if idx:
                d.append(idx)
            else:
                d.append(vocab['<UNK>'])
        if len(d) < max_length:
            for _ in range(max_length - len(d)):
                d.append(vocab['<PAD>'])
        else:
            d = d[:max_length]
        data.append(np.array(d))
    data = np.array(data)
    return data


def preprocess_data_by_split_type(df,split):
    processedData = []
    mp = MeCabParser()
    if split.startswith("root"):
        processedData = df["text"].apply(lambda x: mp.parseByLinearly(hanja.translate(x.strip(), "substitution")) if not isinstance(x, float) else " ").tolist()

        # processedData = [list(MeCabParser().parseByLinearly(hanja.translate(x.strip(), "substitution"))) for x in df["text"] if not isinstance(x, float)]
    elif split.startswith("word"):
        processedData = df["text"].apply(lambda x: list(word_tokenize(hanja.translate(x.strip(), "substitution"))) if not isinstance(x, float) else " ").tolist()
        # processedData = [list(word_tokenize(hanja.translate(x,"substitution"))) for x in df["text"] if not isinstance(x, float)]

    return processedData

def check_pickle_file(pklPath):
    if not os.path.exists(pklPath):
        return False

    return True

def process_make_report(x_test,prediction,raw_text,vocabulary_inv,y_test):
    input_text = [vocabulary_inv.get(str(i)) for i in np.array(x_test).reshape(1, -1)[0]]
    text = [list(filter(lambda x: x != "<PAD>", i)) for i in np.array(input_text).reshape(np.shape(x_test)[0],np.shape(x_test)[1])]
    prediction = (np.array(prediction).reshape(1, -1) + 1)
    label = [list(i).index(1)+1 for i in y_test]

    return list(prediction[0][:len(x_test)]), label, text, list(raw_text)


def process_make_report_no_rawtext(x_test,prediction,raw_text,vocabulary_inv,y_test):
    prediction = (np.array(prediction).reshape(1, -1) + 1)
    label = [list(i).index(1)+1 for i in y_test]

    return list(prediction[0][:len(x_test)]), label


def translated_texts_hanja_to_kor(text):
    return hanja.translate(text,"substitution")


if __name__ == "__main__":
    train_file = './data/train.csv.zip'
    load_data(train_file)