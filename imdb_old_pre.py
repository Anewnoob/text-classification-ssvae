# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.layers.core import Dense
import matplotlib.pyplot as plt
from utils.data_process import init_data
import clstm
from utils.helper import *
import tensorflow.contrib.slim as slim

batch_size=64
iter_num=30#iter_number
embedding_size=250  #embedding size
n_hidden=512 #vae embeddings
c_hidden=512 #classifer embedding
bata=0.5
z_size=50
label_size=2
Train_Size = 2500
un_batch_size = int((40000/Train_Size)*batch_size)  #250
data_path="../data/aclImdb"
class_name = ['negative','positive']
#define the weight and bias dictionary
with tf.name_scope("weight_inital"):
    weights_de={
        'w_':tf.Variable(tf.random_normal([z_size+label_size,n_hidden],mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([2*c_hidden, label_size]))
    }
    biases_de = {
    'b_': tf.Variable(tf.random_normal([n_hidden], mean=0.0, stddev=0.01)),
    'out': tf.Variable(tf.random_normal([label_size]))
    }

#-----------------data procession function------------------------
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x
def extract_character_vocab(total_T):
    special_words = ['<PAD>', '<GO>', '<EOS>']
    set_words = list(set(flatten(total_T)))
    set_words = sorted(set_words)
    set_words = [str(item) for item in set_words]
    print(len(set_words))
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def extract_words_vocab():
    int_to_vocab={idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def getPvector(i):  #Embedding tensor
    return table_X[i]
def get_index(label):
    label = list(set(label))
    label_list = sorted(label)
    return label_list
def get_mask_index(value, label_list):
    return label_list.index(value)
def get_true_index(index, label_list):
    return label_list[index]
def getXs():
    table_X = {}
    vec_path  = '../data/aclImdb/all_%d_vec.txt'%embedding_size
    f_allvec = open(vec_path, 'r')  #
    item = 0
    for line in f_allvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < embedding_size): #delete fist row
            continue
        item += 1  #
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  #
        if lineArr[0] == '</s>':
            table_X['<PAD>']=X  #dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] =X
    f_allvec.close()
    print ("train word number item=", item)
    return table_X
def read_data():
    train_data = list()
    train_label = list()
    test_data = list()
    test_label =list()
    unlabel_data = list()
    label_list = list()

    #train
    ftraindata = open(data_path+'/train_stopword_seq.txt','r')
    train_item = 0
    for line in ftraindata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        train_label.append(int(X[0]))
        train_data.append(X[2:-1])      #del <GO> <EOS>
        train_item += 1
    train_pos = train_data[:int(Train_Size/2)]
    train_neg = train_data[-int(Train_Size/2):]
    train_p_l = train_label[:int(Train_Size/2)]
    train_n_l = train_label[-int(Train_Size/2):]

    train_data = train_pos+train_neg # all tra
    train_label =train_p_l+train_n_l  # all user

    ftraindata.close()

    #test
    ftestdata = open(data_path+'/test_stopword_seq.txt','r')
    test_item = 0
    for line in ftestdata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        test_label.append(int(X[0]))
        test_data.append(X[2:-1])   #del <GO> <EOS>
        test_item += 1
    ftestdata.close()

    #unlable
    funlabeldata = open(data_path+'/train_unlabel_stopword_seq.txt','r')
    unlabel_item = 0
    for line in funlabeldata.readlines():
        lineArr = line.split()
        X = list()
        for i in lineArr:
            X.append(str(i))  # chanage to string or char type
        unlabel_data.append(X[1:-1])  #del <GO> <EOS>
        unlabel_item += 1
    funlabeldata.close()

    #lable_list
    label_list = get_index(train_label)

    print ('train number=', len(train_data),'test number=',len(test_data),'unlabel number = ',len(unlabel_data))
    print ("label numbers=", len(label_list))
    return train_data,train_label,test_data,test_label,unlabel_data,label_list# 返回相关参数
def convert2int(dataset):
    new_dataset = list()
    for i in range(len(dataset)):
        temp = list()
        for j in range(len(dataset[i])):
            temp.append(vocab_to_int[dataset[i][j]])
        new_dataset.append(temp)
    return new_dataset
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
def get_new_table_X(total_data):
    new_table_X= {}
    for i_ in range(len(total_data)):
        for j_ in range(len(total_data[i_])):
            # del low number word and use <PAD> instead of it
            if total_data[i_][j_] not in table_X:
                total_data[i_][j_] = '<PAD>'
            else:
                new_table_X[total_data[i_][j_]] = table_X[total_data[i_][j_]]
    return new_table_X


""" start data processing"""
#init
init_data(data_path,embedding_size)

#get seq,label,label_list,*_table_X
table_X= getXs()
train_data,train_label,test_data,test_label,unlabel_data,label_list=read_data()

#get new_table_X
total_data = train_data + test_data + unlabel_data
new_table_X = get_new_table_X(total_data)

new_table_X['<GO>']=table_X['<GO>']
new_table_X['<EOS>']=table_X['<EOS>']
new_table_X['<PAD>']=table_X['<PAD>']

#get vocab from table_X
voc_tra=list()
for keys in new_table_X:
    voc_tra.append(keys)

#get vocab_to_int
int_to_vocab, vocab_to_int=extract_words_vocab()
#convert to int type
new_train_data = convert2int(train_data)
new_test_data  = convert2int(test_data)
new_unlabel_data = convert2int(unlabel_data)

dic_embeddings = dic_em()
vocab_size = len(dic_embeddings)
print ('Dictionary Size',vocab_size)
""" end data processing"""

W_class_emb = load_class_embedding(vocab_to_int,class_name,dic_embeddings)
dic_embeddings = tf.constant(dic_embeddings)
W_class_emb=tf.constant(W_class_emb)

#-----------------------------------VAE-LOST------------------------------------
def layer_normalization(inputs,
                        epsilon=1e-8,
                        scope="ln",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        outputs = gamma * normalized + beta

    return outputs
def classifer(encoder_embed_input,max_target_sequence_length,keep_prob=0.5,reuse=False):
    with tf.variable_scope("classifier",reuse=reuse):
        with tf.variable_scope("classifier", reuse=reuse):
            encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
            # LSTM
            input_ = tf.transpose(encoder_input, [1, 0, 2])
            fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                        state_is_tuple=True)  # state_is_tuple=True
            fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
            bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0,
                                                        state_is_tuple=True)  # state_is_tuple=True
            bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
            cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32,
                                                                time_major=True)
            # attention
            fw_outputs, bw_outputs = outputs
            W = tf.Variable(tf.random_normal([c_hidden], stddev=0.1))
            h_state = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
            h_state = tf.transpose(h_state, [1, 0, 2])
            M = tf.tanh(h_state)  # M = tanh(h_state)  (batch_size, seq_len, HIDDEN_SIZE)

            alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, c_hidden]),
                                                       tf.reshape(W, [-1, 1])),
                                             (-1, max_target_sequence_length)))  # batch_size x seq_len
            r = tf.matmul(tf.transpose(h_state, [0, 2, 1]),  # (batch,hidden_size,seq_len)
                          tf.reshape(alpha, [-1, max_target_sequence_length, 1]))
            r = tf.squeeze(r)
            l_c = tf.tanh(r)  # (batch , HIDDEN_SIZE)

            l_c = tf.nn.dropout(l_c, keep_prob)
            l_c = layer_normalization(l_c)

            # l_ac = tf.concat([l_c, l_a], 1)  # a [batch a_size]

            FC_W = tf.Variable(tf.truncated_normal([c_hidden, label_size], stddev=0.1))
            #L2
            tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W))
            FC_b = tf.Variable(tf.constant(0., shape=[label_size]))
            pred = tf.nn.xw_plus_b(l_c, FC_W, FC_b)
            return pred
def encoder(encoder_embed_input,y,keep_prob=0.5,reuse=False):
    with tf.variable_scope("encoder",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        encode_lstm = tf.contrib.rnn.LSTMCell(n_hidden,forget_bias=1.0, state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        #new_states = states[-1]  states  tuple  (c,h)

        new_states=tf.concat([states[-1],y],1)
        o_mean = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                   scope="z_mean")
        o_stddev = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                     scope="z_std")
        return outputs, states, o_mean, o_stddev
def decoder(decoder_embed_input,l_y,decoder_y,target_length,max_target_length,encode_state,batch_size,keep_prob,reuse=False):
    with tf.variable_scope("decoder",reuse=reuse):
        #decode_lstm = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        decoder_topic = tf.argmax(l_y,-1)
        decoder_topic = tf.nn.embedding_lookup(W_class_emb, decoder_topic)
        decoder_topic = tf.reshape(decoder_topic,[-1,embedding_size])
        decoder_topic = tf.to_float(decoder_topic)

        decode_lstm = clstm.BasicLSTMCell(n_hidden, label_size=label_size, embedding_size=embedding_size,context=decoder_topic,
                                          forget_bias=1.0, state_is_tuple=True)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        decoder_initial_state = encode_state
        output_layer = Dense(embedding_size) #TOTAL_SIZE
        decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],1)  # add   1  GO to the end
        decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
        decoder_input=tf.concat([decoder_input,decoder_y],2)   #dic_embedding+y(one-hot)
        # # input_=tf.transpose(decoder_input,[1,0,2])
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                            sequence_length=target_length)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, training_helper, decoder_initial_state,
                                                           output_layer)
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                         impute_finished=True,
                                                         maximum_iterations=max_target_length)
        predicting_logits = tf.identity(output.sample_id, name='predictions')
        training_logits = tf.identity(output.rnn_output, 'logits')
        masks = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32, name='masks') #(batch_size,max_target_length)
        #target = tf.concat([target_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  #
        target = decoder_embed_input
        return output,predicting_logits,training_logits,masks,target
def get_cost_c(pred): #compute classifier cost
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=l_y))
    return cost
def get_cost_l(encoder_embed_input,decoder_embed_input,l_y,decoder_y,target_sequence_length,max_target_sequence_length,batch_size,reuse=False):
    encode_outputs, encode_states, z_mean, z_stddev = encoder(encoder_embed_input,l_y, keep_prob,reuse)
    samples = tf.random_normal(tf.shape(z_stddev))
    z = z_mean + tf.exp(z_stddev * 0.5) * samples
    #KL term-------------
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
    latent_cost = tf.reduce_mean(latent_loss)

    l_yz = tf.concat([z, l_y], 1)

    c_state = tf.nn.softplus(tf.matmul(l_yz, weights_de['w_']) + biases_de['b_'])
    #c_state = layer_normalization(c_state)

    decoder_initial_state = clstm.LSTMStateTuple(c_state, encode_states[1])
    decoder_output, predicting_logits, training_logits, masks, target = decoder(decoder_embed_input,l_y,decoder_y,target_sequence_length,max_target_sequence_length,decoder_initial_state,batch_size,keep_prob,reuse)


    #encropy_loss = tf.contrib.seq2seq.sequence_loss(training_logits, target, masks)
    decoder_input=tf.nn.embedding_lookup(dic_embeddings, decoder_embed_input)
    s_loss=tf.square(training_logits-decoder_input)            #batch,len,embeding_size
    mask_loss = tf.reduce_sum(tf.transpose(s_loss, [2, 0, 1]), 0)  # mask_loss (bacth_size,max_len_seq)
    encropy_loss = tf.reduce_mean(tf.multiply(mask_loss, masks), 1)  #还原句子长度 其余位置都是0　　multiply　点乘
    cost = tf.add(encropy_loss, (latentscale_iter * (latent_loss)))   #cost  (batch_size)

    return cost

def get_cost_l_all(encoder_embed_input,decoder_embed_input,l_y,decoder_y,target_sequence_length,max_target_sequence_length,batch_size,reuse=False):
    flag = 0
    real_label_cost = get_cost_l(encoder_embed_input, decoder_embed_input, l_y, decoder_y, target_sequence_length,
                                 max_target_sequence_length, batch_size, reuse=False)
    for label in range(label_size):
        y_i = get_onehot(label)
        batch_y = [y_i] * batch_size
        if batch_y != l_y:
            wrong_label_cost = get_cost_l(encoder_embed_input, decoder_embed_input, batch_y, vae_y_all[label], target_sequence_length,
                       max_target_sequence_length, batch_size, reuse=True)
            wrong_cost = tf.expand_dims(wrong_label_cost, 1)
            if flag == 0:
                wrong_cost_all = tf.identity(wrong_cost)
                flag = 1
            else:
                wrong_cost_all = tf.concat([wrong_cost_all, wrong_cost],1)
    wrong_cost_mean = tf.reduce_mean(wrong_cost_all,1)
    vae_pred_index = tf.argmin(wrong_cost_all, 1)
    vae_y = tf.one_hot(vae_pred_index, label_size)
    label_0 = tf.slice(wrong_cost_all,[0,0],[-1,1])
    label_1 = tf.slice(wrong_cost_all,[0,1],[-1,1])
    new_wrong_cost = tf.concat([label_1,label_0],1)

    vae_pred_y = tf.nn.softmax(new_wrong_cost)
    cost_vae = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vae_pred_y, labels=l_y))
    same_pred = tf.equal(tf.argmax(vae_pred_y, 1), tf.argmax(l_y, 1))
    accuracy = tf.reduce_mean(tf.cast(same_pred, tf.float32))

    target_cost = real_label_cost-0.1*wrong_cost_mean+cost_vae
    return tf.reduce_mean(target_cost),real_label_cost,wrong_cost_all,accuracy

def get_cost_u(u_encoder_embed_input,u_decoder_embed_input):
    prob_y=classifer(u_encoder_embed_input,un_max_target_sequence_length-1,keep_prob=keep_prob,reuse=True)
    prob_y=tf.nn.softmax(prob_y)
    for label in range(label_size):
        y_i=get_onehot(label)
        cost_l =get_cost_l(u_encoder_embed_input,u_decoder_embed_input,[y_i]*un_batch_size,vae_y_u[label],un_target_sequence_length,un_max_target_sequence_length,un_batch_size,reuse=True)
        u_cost = tf.expand_dims(cost_l, 1)  # 在cost_l张量中的第１个位置（从０开始）增加一个维度 cost_l (batch) u_cost(batch,1)
        if label == 0:
            L_ulab = tf.identity(u_cost)
        else:
            L_ulab = tf.concat([L_ulab, u_cost],1)  #累加整个label_size中的值的loss
    un_pred_index = tf.argmin(L_ulab, 1)
    un_vae_y = tf.one_hot(un_pred_index, label_size)

    un_target_y = prob_y+un_vae_y*(0.5/alpha)  #0.1 0.3 0.5 0.7 1.0
    un_target_y= tf.nn.softmax(un_target_y)
    #un_target_y = tf.argmax(un_target_y, 1)
    #un_target_y = tf.one_hot(un_target_y, label_size)
    same_pred = tf.equal(tf.argmax(prob_y, 1), tf.argmax(un_vae_y, 1))
    accuracy = tf.reduce_mean(tf.cast(same_pred, tf.float32))
    un_cost_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob_y, labels=un_vae_y))
    U = tf.reduce_sum(tf.multiply(L_ulab, prob_y),1) #- tf.multiply(prob_y, tf.log(prob_y)))  #    U(batch)
    return U,L_ulab,prob_y,accuracy,un_cost_c

def creat_y_scopus(label_y,seq_length): #label data
    lcon_y= [label_y for j in range(seq_length)]
    return lcon_y
def creat_u_y_scopus(seq_length): #unlabel data
    ucon_y=[]
    for i in range(label_size):
        label_y=get_onehot(i)
        temp=[]
        for j in range(un_batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        ucon_y.append(temp)
    return ucon_y
def creat_all_y_scopus(seq_length): #unlabel data
    all_con_y=[]
    for i in range(label_size):
        label_y=get_onehot(i)
        temp=[]
        for j in range(batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        all_con_y.append(temp)
    return all_con_y

#tensor definition
keep_prob = tf.placeholder("float")
training = tf.placeholder(tf.bool)
alpha = tf.placeholder("float")
c_alpha = tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
input_x = tf.placeholder(dtype=tf.int32)
l_y=tf.placeholder(dtype=tf.float32,shape=[batch_size,label_size])
vae_y=tf.placeholder("float",[batch_size,None,label_size])  #vae_y
vae_y_all=tf.placeholder("float",[label_size,batch_size,None,label_size])
vae_y_u=tf.placeholder("float",[label_size,un_batch_size,None,label_size])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
u_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None])
u_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None])
latentscale_iter=tf.placeholder(dtype=tf.float32)


target_cost_l,cost_l,wrong_cost_l,l_acc = get_cost_l_all(l_encoder_embed_input,l_decoder_embed_input,l_y,vae_y,target_sequence_length,max_target_sequence_length,batch_size)

cost = tf.reduce_mean(target_cost_l)
optimizer=tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)




def train_model():
    #exclude = ['classifier','classifier','Variable_1','Adam']
    #exclude = ['classifier', 'classifier', 'ln', 'Variable_1', 'Adam_1']
    #exclude = ['classifier','classifier','bidirectional_rnn','bw','multi_rnn_cell','cell_0','basic_lstm_cell','kernel','Adam_1']
    #variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    saver=tf.train.Saver()
    initial = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as  sess:
        sess.run(initial)
        #saver.restore(sess, './temp/imdb_2500_pre_l_new_512.pkt')
        print('Read train & test data')
        initial_learning_rate = 0.0004
        learning_rate_len = 0.000008
        min_kl=0.0
        min_kl_epoch=min_kl #退火参数
        kl_lens = 0.008

        #sort for train dataset
        new_trainS,new_trainL = sort_dataset(new_train_data,train_label)

        #-----------------------------------
        tempU = list(set(label_list))
        TRAIN_DIC = {}
        for i in range(len(tempU)):
            TRAIN_DIC[i] = [0, 0, 0]  # use mask

        count = 0
        alpha_epoch=1
        alpha_value=(2.0-1.0)/iter_num
        train_cost_list = []


        for epoch in range(iter_num):
            #initial_learning_rate -= learning_rate_len
            if(initial_learning_rate<=0):
                initial_learning_rate=0.000001
            step=0
            acc=0
            train_cost=0
            label_cost=0
            unlabel_cost=0
            classifier_cost=0
            while step < len(new_trainS) // batch_size:  #2500/25
                start_i = step * batch_size
                input_x = new_trainS[start_i:start_i + batch_size]


                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])  #padding the same(max lenth)
                encode_batch=eos_sentence_batch(input_x,vocab_to_int['<EOS>'])      #append <EOS> to seq
                input_batch=pad_sentence_batch(encode_batch,vocab_to_int['<PAD>'])  #padding

                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source)+1)   #+1

                target_maxlength=len(input_batch[0])+1 #get max length  +1


                if min_kl_epoch<1.0:
                    min_kl_epoch = min_kl + count* kl_lens
                else:
                    min_kl_epoch=1.0
                batch_y = []
                decode_y=[]
                user_mask_id = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainL[y_i], label_list))
                    user_mask_id.append(get_mask_index(new_trainL[y_i], label_list))
                    TRAIN_DIC.get(get_mask_index(new_trainL[y_i], label_list))[2]+=1 #Groud value Groud Truth a+c
                    decode_y.append(creat_y_scopus(xsy_step,target_maxlength)) #copy
                    batch_y.append(xsy_step)

                decode_y_all = creat_all_y_scopus(target_maxlength)
                #acc_un, ulab, p_y,c_pred,batch_cost,c_cost,u_cost
                acc_l,wrong_l_cost,op,batch_cost,l_cost=sess.run([l_acc,wrong_cost_l,optimizer,cost,cost_l],
                                                                              feed_dict={vae_y:decode_y,vae_y_all:decode_y_all,
                                                                              l_encoder_embed_input:sources_batch, l_decoder_embed_input:input_batch,
                                                                              l_y:batch_y,

                                                                              it_learning_rate: initial_learning_rate,latentscale_iter:min_kl_epoch,
                                                                              keep_prob: 0.5,target_sequence_length: pad_source_lengths,

                                                                            })
                if(step%10==0 and step is not 0):
                    print ('min_kl_epoch',min_kl_epoch)
                    print ('TRAIN LOSS', train_cost, 'LABEL COST', label_cost, 'Unlabel Cost', unlabel_cost, 'Classifier Cost', classifier_cost)
                    print ("\n")
                loss=np.mean(batch_cost)
                lbatch_cost=np.mean(l_cost)

                label_cost+=lbatch_cost
                train_cost+=loss
                train_cost_list.append(train_cost)

                step+=1 # while
                count+=1
            print ("acc_l",acc_l,"l_cost",l_cost,"lab",wrong_l_cost,"y",batch_y)
            # print "acc_un:",acc_un,"ulab",ulab,"p_y",p_y
            alpha_epoch+=alpha_value

            # Precision Recall, F1
            P = []
            R = []
            for i in TRAIN_DIC.keys():
                # print TRAIN_DIC.get(i)[0],TRAIN_DIC.get(i)[1]
                if TRAIN_DIC.get(i)[1] == 0:
                    TRAIN_DIC.get(i)[1] = 1
                if TRAIN_DIC.get(i)[2] == 0:
                    TRAIN_DIC.get(i)[2] = 1
                Pi = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[1]
                Ri = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[2]
                P.append(Pi)
                R.append(Ri)

            print ('\nTRAIN RESULT')

            print ('total train number', step * batch_size, 'learning rate', initial_learning_rate)
            print ('iter', epoch, 'Accuracy', acc / (step * batch_size),'TRAIN LOSS', train_cost)

        model_path = './temp/imdb_%d_pre_512_50_topic.pkt' % Train_Size
        saver.save(sess, model_path)



def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
def pad_sentence_batch(sentence_batch, pad_int,max_len = 400):
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def sort_dataset(dataset,label = None):
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
    return new_data,new_label


if __name__ == "__main__":
    train_model()
    print ('------------Model END------------')