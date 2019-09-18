# -*- coding:  UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def save_metrics(LEARN_RATE, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, root='result/bk_metric_vae.txt'):
    files = open(root, 'a+')
    files.write('epoch \t learning_rate \t Precision \t Recall \t F1 \t ACC1 \n')
    for i in range(len(TRAIN_P)):
        files.write(str(i) + '\t')
        files.write(str(LEARN_RATE[i]) + '\t')
        files.write(str(TRAIN_P[i]) + '\t' + str(TRAIN_R[i]) + '\t' + str(TRAIN_F1[i]) + '\t' + str(
            TRAIN_ACC1[i]) + '\t' + '\n')
    files.close()


def draw_pic_metric(P, R, F1, ACC1, name='train'):
    font = {'family': name,
            'weight': 'bold',
            'size': 18
            }
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    train_axis = np.array(range(1, len(P) + 1, 1))
    plt.plot(train_axis, np.array(P), "b--", label="P")
    train_axis = np.array(range(1, len(R) + 1, 1))
    plt.plot(train_axis, np.array(R), "r--", label="R")
    train_axis = np.array(range(1, len(F1) + 1, 1))
    plt.plot(train_axis, np.array(F1), "g--", label="F1-score")
    train_axis = np.array(range(1, len(ACC1) + 1, 1))
    plt.plot(train_axis, np.array(ACC1), "y--", label="ACC")

    plt.title(name)
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('value')
    plt.xlabel('Training iteration')
    plt.show()

def creat_y_scopus(label_y, seq_length):  # label data
    lcon_y = [label_y for j in range(seq_length)]
    return lcon_y

def get_mask_index(value, label_list):
    return label_list.index(value)

def eos_sentence_batch(sentence_batch, eos_in):
    return [sentence + [eos_in] for sentence in sentence_batch]  #


def pad_sentence_batch(sentence_batch, pad_int, max_len=512):
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    #max_sentence = max_len
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    # max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    # if max_sentence <= max_len:
    #     return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    # else:
    #     new_sentence = []
    #     for sentence in sentence_batch:
    #         if len(sentence)<max_len:
    #             sentence = sentence+[pad_int]*(max_len - len(sentence))
    #             new_sentence.append(sentence)
    #         else:
    #             sentence = sentence[:max_len]
    #             new_sentence.append(sentence)
    #     return new_sentence

def load_class_embedding(vocab_to_int,class_name,dic_embeddings):
    print("load class embedding...")
    name_list = [ k.lower().split(' ') for k in class_name]
    id_list = [ [ vocab_to_int[i] for i in l] for l in name_list]
    value_list = [ [ dic_embeddings[i] for i in l]    for l in id_list]
    value_mean = [ np.mean(l,0)  for l in value_list]
    return np.asarray(value_mean)
