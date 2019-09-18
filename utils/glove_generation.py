# -*- coding:  UTF-8 -*-
import numpy as np
import cPickle  as pkl
import gensim
import shutil
from sys import platform


# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load_modle(filename):
    num_lines = getFileLineNums(filename)
    print "num_lines:",num_lines
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    print gensim_first_line
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    print "start loading model"
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model


def load_embedding_vectors_glove_gensim(vocabulary, filename):
    print("loading embedding")
    #model = load_modle(filename)
    model = gensim.models.KeyedVectors.load_word2vec_format("glove_model.txt")
    print "load model succeesful!"
    vector_size = model.vector_size
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    glove_vocab = list(model.vocab.keys())
    count = 0
    mis_count = 0
    for word in vocabulary.keys():
        print word
        idx = vocabulary.get(word)
        if word in glove_vocab:
            embedding_vectors[idx] = model.wv[word]
            count += 1
        else:
            mis_count += 1
    print("num of vocab in glove: {}".format(count))
    print("num of vocab not in glove: {}".format(mis_count))
    return embedding_vectors

if __name__ == '__main__':
    embedding_size = 300
    Label_Train_Size = 2500
    Unlabel_Train_Size = 40000
    embpath = "../../data/aclImdb/imdb_full_glove.p"
    target_pkl_path = "../../data/aclImdb/imdb_%d_%d_%d.pkl" % (Label_Train_Size, Unlabel_Train_Size, embedding_size)
    # x = cPickle.load(open(loadpath, "rb"))
    # train, val, test = x[0], x[1], x[2]
    # train_lab, val_lab, test_lab = x[3], x[4], x[5]
    # wordtoix, ixtoword = x[6], x[7]
    data = pkl.load(open(target_pkl_path, "rb"))
    dic_embeddings = data['dic_embeddings']
    new_unlabelS = data['new_unlabelS']
    new_testS = data['new_testS']
    new_trainS = data['new_trainS']
    int_to_vocab = data['int_to_vocab']
    vocab_to_int = data['vocab_to_int']
    new_trainL = data['train_label']
    new_testL = data['test_label']
    label_list = data['label_list']
    print "load data pkl successful!"

    print("load data finished")

    glove_path = '../../data/aclImdb/glove.840B.300d.txt'
    y = load_embedding_vectors_glove_gensim(vocab_to_int,glove_path )
    pkl.dump([y.astype(np.float32)], open(embpath, 'wb'))

