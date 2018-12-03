# -*- coding: utf-8 -*-
import os
import sys
from nltk.corpus import stopwords
import nltk

reload(sys)
sys.setdefaultencoding('utf8')


def load_train(path):
    print("load train data...")
    if not os.path.exists(path + '/pos_all.txt') or not os.path.exists(path + '/neg_all.txt'):
        if os.path.exists(path + '/pos_all.txt'):
            os.remove(path + '/pos_all.txt')
        if os.path.exists(path + '/neg_all.txt'):
            os.remove(path + '/neg_all.txt')
        merge_file(path)
    pos_list, neg_list, label = read_file(path)
    pos_list.extend(neg_list)

    #stopwords symbol
    seq_list = handle_data(pos_list)
    return seq_list,label
def load_test(path):
    print("load test data...")
    if not os.path.exists(path + '/pos_all.txt') or not os.path.exists(path + '/neg_all.txt'):
        if os.path.exists(path + '/pos_all.txt'):
            os.remove(path + '/pos_all.txt')
        if os.path.exists(path + '/neg_all.txt'):
            os.remove(path + '/neg_all.txt')
        merge_file(path)
    pos_list,neg_list,label = read_file(path)
    pos_list.extend(neg_list)

    #stopwords symbol
    seq_list = handle_data(pos_list)
    return seq_list,label
def load_unsup(path):
    if not os.path.exists(path + '/un_all.txt'):
        dir_path = path + "/unsup"
        '''merge'''
        files = os.listdir(dir_path)
        unlabel_all = open(path + '/un_all.txt','a')
        all = []
        for file in files:
            whole_location = os.path.join(dir_path, file)
            with open(whole_location, 'r') as f:
                line = f.readlines()
                all.extend(line)
        for file in all:
            unlabel_all.write(file)
            unlabel_all.write('\n')
        unlabel_all.close()

    print("load unlabel data...")
    '''read data'''
    un_list = []
    with open(path + '/un_all.txt', 'r')as f:
        line = f.readlines()
        for seq in line:
            seq = seq.strip("\n")
            un_list.append(seq)

    #stopwords symbol
    seq_list = handle_data(un_list)
    return seq_list
def load_wdict(path):
    wdict_list = []
    wdict_path = path + '/imdb.vocab'
    with open(wdict_path, "r") as f:
        lines = f.readlines()
        for word in lines:
            word = word.strip("\n")
            wdict_list.append(word)
    return wdict_list

def merge_file(path):
    '''merge the file'''
    pos_location = path + '/pos'
    pos_files = os.listdir(pos_location)
    neg_location = path + '/neg'
    neg_files = os.listdir(neg_location)
    pos_all = open(path + '/pos_all.txt', 'a')
    neg_all = open(path + '/neg_all.txt', 'a')
    all = []
    for file in pos_files:
        whole_location = os.path.join(pos_location, file)
        with open(whole_location, 'r') as f:
            line = f.readlines()
            all.extend(line)
    for file in all:
        pos_all.write(file)
        pos_all.write('\n')
    alls = []
    for file in neg_files:
        whole_location = os.path.join(neg_location, file)
        with open(whole_location, 'r') as f:
            line = f.readlines()
            alls.extend(line)
    for file in alls:
        neg_all.write(file)
        neg_all.write('\n')
    pos_all.close()
    neg_all.close()
def merge_all(path):
    '''merge all'''
    all =[]
    with open(path+"/train_stopword_seq.txt",'r') as f:
        line = f.readlines()
        all.extend(line)
    with open(path+"/train_unlabel_stopword_seq.txt",'r') as f:
        line = f.readlines()
        all.extend(line)
    with open(path+"/test_stopword_seq.txt",'r') as f:
        line = f.readlines()
        new_line = []
        for i in line:
            i = i[2:]# del label
            new_line.append(i)
        all.extend(new_line)
    with open(path+"/all_stopword_seq.txt",'a') as f:
        for seq in all:
            f.write(seq)

def read_file(path):
    '''read data from file'''
    pos_list = []
    with open(path + '/pos_all.txt', 'r')as f:
        line = f.readlines()
        for seq in line:
            seq = seq.strip("\n")
            pos_list.append(seq)
        #pos_list.extend(line)
    neg_list = []
    with open(path + '/neg_all.txt', 'r')as f:
        line = f.readlines()
        for seq in line:
            seq = seq.strip("\n")
            neg_list.append(seq)

    # label
    label = [1 for i in range(12500)]
    label.extend([0 for i in range(12500)])

    return pos_list,neg_list,label
def handle_data(data):
    '''del the word or simbol don't need'''
    #stopwords symbol
    seq = []        #nD  [[],[],[],[],[],...,[]]
    #seqtence = []   #1D  []
    #nltk.download()
    stop_words = set(stopwords.words('english'))
    symbol = set(stopwords.words('symbol'))
    for con in data:
        con = con.decode("utf-8")
        words = nltk.word_tokenize(con)
        line = []
        for word in words:
            word = word.lower()
            if word.isalpha() and word not in stop_words and word not in symbol:
                line.append(word)
        seq.append(line)
    return seq
def write_data(path,seq_list,label = None):
    if not os.path.exists(path + '_stopword_seq.txt'):
        with open(path + '_stopword_seq.txt','a')as f:
            for index,seq in enumerate(seq_list):
                seq = [str(i) for i in seq]
                s = '<GO> ' + " ".join(seq) + ' <EOS>'
                if label:
                    s = str(label[index]) +" "+s
                f.write(s)
                f.write('\n')


#main
def init_data(data_path,embeddings_size):
    train_path = data_path + "/train"
    test_path = data_path + "/test"
    unlabel_path = train_path


    if not os.path.exists(data_path+"/train_stopword_seq.txt") or not os.path.exists(data_path+"/train_unlabel_stopword_seq.txt") or not os.path.exists(data_path+"/test_stopword_seq.txt"):
        if os.path.exists(data_path+"/train_stopword_seq.txt"):
            os.remove(data_path+"/train_stopword_seq.txt")
        if os.path.exists(data_path+"/train_unlabel_stopword_seq.txt"):
            os.remove(data_path+"/train_unlabel_stopword_seq.txt")
        if os.path.exists(data_path+"/test_stopword_seq.txt"):
            os.remove(data_path+"/test_stopword_seq.txt")
        #write and load data set
        train_x,train_y = list(load_train(train_path))       #tuple to list
        test_x,test_y = list(load_test(test_path))
        unlabel = load_unsup(unlabel_path)
        #wdict = load_wdict(wdict_path)

        print "write..."
        write_data(train_path, train_x, train_y)
        write_data(test_path, test_x, test_y)
        write_data(unlabel_path+"_unlabel", unlabel)
        if os.path.exists(data_path+"/all_stopword_seq.txt"):
            os.remove(data_path+"/all_stopword_seq.txt")
        merge_all(data_path)
    else:
        if not os.path.exists(data_path+"/all_stopword_seq.txt"):
            print "merge all..."
            merge_all(data_path)

    if not os.path.exists(data_path+"/all_250_vec.txt"):
        #word2ec
        print "word2vec..."
        command = "word2vec -train all_stopword_seq.txt -output all_%d_vec.txt -size %d -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3"%(embeddings_size,embeddings_size)
        os.system("cd %s;%s"%(data_path,command))
    print "-------------data process down------------"

#init_data(data_path="../../data/aclImdb",embeddings_size = 250)
