# -*- coding:  UTF-8 -*-
import os
import pickle  as pkl
from Imdb.utils.data_process import init_data
import collections

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def extract_character_vocab(total_data):
    special_words = ['<PAD>', '<UNK>','<GO>', '<EOS>']
    set_words = list(set(flatten(total_data)))
    set_words = sorted(set_words)
    set_words = [str(item) for item in set_words]
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def extract_words_vocab(voc_tra):
    int_to_vocab = {idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def get_index(label):
    label = list(set(label))
    label_list = sorted(label)
    return label_list

def getXs(embedding_size):
    table_X = {}
    vec_path = '../data/aclImdb/all_%d_vec.txt' % embedding_size
    f_allvec = open(vec_path, 'r')  #
    item = 0
    for line in f_allvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < embedding_size):  # delete fist row
            continue
        item += 1  #
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  #
        if lineArr[0] == '</s>':
            table_X['<PAD>'] = X  # dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] = X
    f_allvec.close()
    print ("word sum=", item)
    return table_X

def read_data(data_path,Label_Train_Size):
    train_data = list()
    train_label = list()
    test_data = list()
    test_label = list()
    unlabel_data = list()

    # train
    ftraindata = open(data_path + '/train_stopword_seq.txt', 'r')
    train_item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        train_label.append(int(X[0]))
        train_data.append(X[2:-1])  # del <GO> <EOS> label
        train_item += 1
    train_pos = train_data[:int(Label_Train_Size / 2)]
    train_neg = train_data[-int(Label_Train_Size / 2):]
    train_p_l = train_label[:int(Label_Train_Size / 2)]
    train_n_l = train_label[-int(Label_Train_Size / 2):]

    train_data = train_pos + train_neg  # all tra
    train_label = train_p_l + train_n_l  # all user

    ftraindata.close()

    # test
    ftestdata = open(data_path + '/test_stopword_seq.txt', 'r')
    test_item = 0
    for line in ftestdata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        test_label.append(int(X[0]))
        test_data.append(X[2:-1])  # del <GO> <EOS> label
        test_item += 1
    ftestdata.close()

    # unlable
    funlabeldata = open(data_path + '/train_unlabel_stopword_seq.txt', 'r')
    unlabel_item = 0
    for line in funlabeldata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        unlabel_data.append(X[1:-1])  # del <GO> <EOS>
        unlabel_item += 1
    funlabeldata.close()

    # lable_list
    label_list = get_index(train_label)

    print ('train number=', len(train_data), 'test number=', len(test_data), 'unlabel number = ', len(unlabel_data))
    print ("label numbers=", len(label_list))
    return train_data, train_label, test_data, test_label, unlabel_data, label_list

def convert2int(dataset,vocab_to_int):
    new_dataset = list()
    for i in range(len(dataset)):
        temp = list()
        for j in range(len(dataset[i])):
            index = vocab_to_int.get(dataset[i][j],vocab_to_int['<UNK>'])
            temp.append(index)
        new_dataset.append(temp)
    return new_dataset

def dic_em(new_table_X):
    dic_embeddings = list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings

def get_new_table_X(total_data,table_X):
    new_table_X = {}
    for i_ in range(len(total_data)):
        for j_ in range(len(total_data[i_])):
            # del low number word and use <PAD> instead of it
            if total_data[i_][j_] not in table_X:
                total_data[i_][j_] = '<UNK>'
            else:
                new_table_X[total_data[i_][j_]] = table_X[total_data[i_][j_]]
    new_table_X['<GO>'] = table_X['<GO>']
    new_table_X['<PAD>'] = table_X['<PAD>']
    new_table_X['<UNK>'] = table_X['<PAD>']
    new_table_X['<EOS>'] = table_X['<EOS>']
    return new_table_X

def maxlen_dataset(dataset,label=None,max_len = 512):
    new_data = []
    new_label = []
    # for index,sentence in enumerate(dataset):
    #     sentence_len = len(sentence)
    #     if sentence_len > max_len:
    #         split_num = 1
    #         s_len = sentence_len
    #         while s_len > max_len:
    #             split_num+=1
    #             s_len = sentence_len // split_num
    #         #split_num = sentence_len // max_len + 1
    #         #print (sentence_len,s_len,split_num)
    #         for i in range(split_num):
    #             start_i = i * s_len
    #             if start_i+max_len>sentence_len:
    #                 new_data.append(sentence[start_i:sentence_len])
    #                 if label:
    #                     new_label.append(label[index])
    #             new_data.append(sentence[start_i:start_i + s_len])
    #             if label:
    #                 new_label.append(label[index])
    #     new_data.append(sentence)
    #     if label:
    #         new_label.append(label[index])
    for index, sentence in enumerate(dataset):
        sentence_len = len(sentence)
        if sentence_len > max_len:
            new_data.append(sentence[:max_len])
        else:
            new_data.append(sentence)
        if label:
            new_label.append(label[index])
    return new_data,new_label


def sort_dataset(dataset, label=None):
    index_T = {}
    new_data = []
    new_label = []
    for i in range(len(dataset)):
        index_T[i] = len(dataset[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        new_data.append(dataset[id])
        if label:
            new_label.append(label[id])
    return new_data, new_label

def dump_data(data_path,embedding_size,Label_Train_Size):

    """ start data processing"""
    # init
    init_data(data_path, embedding_size)

    # get seq,label,label_list,*_table_X
    table_X = getXs(embedding_size)
    train_data, train_label, test_data, test_label, unlabel_data, label_list = read_data(data_path,Label_Train_Size)

    # get new_table_X
    total_data = train_data + test_data + unlabel_data

    #int_to_vocab, vocab_to_int = extract_character_vocab(total_data)
    new_table_X = get_new_table_X(total_data,table_X)

    # get vocab from table_X
    voc_tra = list()
    for keys in new_table_X:
        voc_tra.append(keys)

    # get vocab_to_int
    int_to_vocab, vocab_to_int = extract_words_vocab(voc_tra)

    # convert to int type
    new_train_data = convert2int(train_data,vocab_to_int)
    new_test_data = convert2int(test_data,vocab_to_int)
    new_unlabel_data = convert2int(unlabel_data,vocab_to_int)
    #print (len(new_unlabel_data))

    #max_len = 512
    new_train_data,train_label = maxlen_dataset(new_train_data, train_label, max_len=800)
    #new_test_data,test_label = maxlen_dataset(new_test_data, test_label, max_len=320)
    new_unlabel_data,_ = maxlen_dataset(new_unlabel_data, max_len=400)
    print (len(new_unlabel_data))

    # sort for train dataset
    new_trainS, new_trainL = sort_dataset(new_train_data, train_label)
    new_testS, new_testL = sort_dataset(new_test_data, test_label)
    new_unlabelS, _ = sort_dataset(new_unlabel_data)

    dic_embeddings = dic_em(new_table_X)
    vocab_size = len(dic_embeddings)
    print ('Dictionary Size', vocab_size)


    del train_data, test_data, unlabel_data, table_X, total_data, \
        new_train_data, new_test_data, new_unlabel_data, train_label, test_label

    #pkl
    key = ['dic_embeddings','new_unlabelS','new_testS','new_trainS','int_to_vocab','vocab_to_int','train_label','test_label','label_list']
    value = [dic_embeddings,new_unlabelS,new_testS,new_trainS,int_to_vocab,vocab_to_int,new_trainL,new_testL,label_list]
    data = dict(zip(key,value))
    print("dump into file imdb.pkl...")
    pkl_path = "../data/aclImdb/imdb_%d_%d_800.pkl"%(Label_Train_Size,embedding_size)
    if os.path.exists(pkl_path):
        os.mkdir(pkl_path)
    pkl.dump(data, open(pkl_path , "wb"))
    print ("dump data success! file_path:%s"%pkl_path)

    """ end data processing"""
    return dic_embeddings,new_unlabelS,new_testS,new_trainS,int_to_vocab,vocab_to_int,new_trainL,new_testL,label_list