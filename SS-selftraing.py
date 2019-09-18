# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.layers.core import Dense
from compiler.ast import flatten
import matplotlib.pyplot as plt
from utils.data_process import init_data


#----------------------------------config-----------------------
batch_size=64

iter_num=30#iter_number
embedding_size=250  #embedding size
c_hidden=512 #classifer embedding
label_size=2
Train_Size = 2500
initial_learning_rate = 0.0004
un_batch_size = int((2500/Train_Size)*batch_size)  #100
data_path="../data/aclImdb"

#-----------------data procession function------------------------
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x
def extract_words_vocab():
    print ('dictionary length',len(voc_tra))
    int_to_vocab={idx: word for idx, word in enumerate(voc_tra)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def get_index(label):
    label = list(set(label))
    label_list = sorted(label)
    return label_list
def get_mask_index(value, label_list):
    return label_list.index(value)
def getXs():
    table_X = {}
    f_allvec = open('../data/aclImdb/all_250_vec.txt', 'r')  #
    item = 0
    for line in f_allvec.readlines():
        lineArr = line.split()
        if (len(lineArr) < 250): #delete fist row
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
def creat_y_scopus(label_y,seq_length): #label data
    lcon_y= [label_y for j in range(seq_length)]
    return lcon_y
def creat_u_y_scopus(seq_length): #unlabel data
    ucon_y=[]
    for i in range(label_size):
        label_y=get_onehot(i)
        temp=[]
        for j in range(batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        ucon_y.append(temp)
    return ucon_y
def save_metrics(LEARN_RATE,TRAIN_P,TRAIN_R,TRAIN_F1,TRAIN_ACC1,root='result/self-training.txt'):
    files=open(root,'a+')
    files.write('epoch \t learning_rate \t Precision \t Recall \t F1 \t ACC1 \n')
    for i in range(len(TRAIN_P)):
        files.write(str(i)+'\t')
        files.write(str(LEARN_RATE[i])+'\t')
        files.write(str(TRAIN_P[i]) + '\t'+str(TRAIN_R[i])+'\t'+str(TRAIN_F1[i])+'\t'+str(TRAIN_ACC1[i])+'\t'+'\n')
    files.close()
def draw_pic_metric(P,R,F1,ACC1,name='train'):
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
    plt.xlabel('iteration')
    plt.show()
def pad_sentence_batch(sentence_batch, pad_int):
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
print ('Dictionary Size',len(dic_embeddings))
""" end data processing"""


"""tensor definition"""
dic_embeddings=tf.constant(dic_embeddings)
training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder("float")
alpha = tf.placeholder("float")
it_learning_rate=tf.placeholder("float")
l_y=tf.placeholder(dtype=tf.float32,shape=[batch_size,label_size],name = 'label')
#un_y=tf.placeholder(dtype=tf.float32,shape=[un_batch_size,label_size])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='un_target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='un_max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None],name = 'l_dec_input')
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None],name = 'l_enc_input')
u_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None],name = 'u_enc_input')
u_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None],name = 'u_dec_input')
latentscale_iter=tf.placeholder(dtype=tf.float32)


#-----------------------------------Classifer------------------------------------
def get_cost_c(pred,l_y): #compute classifier cost
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=l_y))
    return cost
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
    with tf.variable_scope("classifier", reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        # LSTM
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        fw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0, state_is_tuple=True)  # state_is_tuple=True
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        bw_lstm_cell = tf.contrib.rnn.BasicLSTMCell(c_hidden, forget_bias=1.0, state_is_tuple=True)  # state_is_tuple=True
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)  # 加入dropout
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32, time_major=True)
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

        FC_W = tf.Variable(tf.truncated_normal([c_hidden, label_size], stddev=0.1))
        tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W))
        FC_b = tf.Variable(tf.constant(0., shape=[label_size]))
        pred = tf.nn.xw_plus_b(l_c, FC_W, FC_b)
        return pred




pred=classifer(l_encoder_embed_input,max_target_sequence_length,keep_prob =keep_prob)
un_pred=classifer(u_encoder_embed_input,un_max_target_sequence_length,keep_prob =keep_prob,reuse = True)
un_pred_index = tf.argmax(un_pred,1)
un_y = tf.one_hot(un_pred_index,label_size)

cost_c=get_cost_c(pred,l_y)
tf.add_to_collection("losses_c", cost_c)
cost_c = tf.add_n(tf.get_collection("losses_c"))

un_cost_c = get_cost_c(un_pred,un_y)
cost = cost_c #+ alpha*un_cost_c


pred = tf.nn.softmax(pred)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(l_y,1))
correct_pred_f = tf.cast(correct_pred, tf.int32)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

tvars = tf.trainable_variables()
global_step = tf.Variable(0, name="global_step", trainable=False)
gradients = tf.gradients(cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step,
                                               name='train_step')



def train_model():
    saver=tf.train.Saver()
    initial = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initial)
        #saver.restore(sess, './temp/bk_tul_lvae.pkt')
        print('----------Read train & test data--------------')

        #sort for train dataset
        new_trainS,new_trainL = sort_dataset(new_train_data,train_label)
        new_testS, new_testL = sort_dataset(new_test_data, test_label)
        new_unlabelS,_= sort_dataset(new_unlabel_data)
        #-----------------------------------

        tempU = list(set(label_list))
        TRAIN_DIC = {}
        for i in range(len(tempU)):
            TRAIN_DIC[i] = [0, 0, 0]  # use mask
        TRAIN_P = []
        TRAIN_R = []
        TRAIN_F1 = []
        TRAIN_ACC1 = []

        TEST_P = []
        TEST_R = []
        TEST_F1 = []
        TEST_ACC1 = []
        Learning_rate = []
        count = 0
        train_cost_list = []
        T1 = 2
        T2 = 10
        alpha_u = 0.
        af = 0.3
        for epoch in range(iter_num):
            step=0
            acc=0
            train_cost=0
            classifier_cost=0
            if epoch > T1:
                alpha_u = (epoch-T1) / (T2-T1) * af
                if epoch > T2:
                    alpha_u = af
            while step < len(new_trainS) // batch_size:
                start_i = step * batch_size
                un_start_i = step * un_batch_size
                input_x = new_trainS[start_i:start_i + batch_size]
                input_ux=new_unlabelS[un_start_i:un_start_i + un_batch_size]

                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                un_sources_batch = pad_sentence_batch(input_ux, vocab_to_int['<PAD>'])


                #unlabel
                un_pad_source_lengths = []
                for source in input_ux:
                    un_pad_source_lengths.append(len(source))  # +1 is add <GO> word lenth in decoder
                #label
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source))   #+1


                batch_y = []
                user_mask_id = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainL[y_i], label_list))
                    user_mask_id.append(get_mask_index(new_trainL[y_i], label_list))
                    TRAIN_DIC.get(get_mask_index(new_trainL[y_i], label_list))[2]+=1 #Groud value Groud Truth a+c
                    batch_y.append(xsy_step)

                un_pred_batch,pred_batch,c_pred,op,batch_cost,c_cost=sess.run([un_pred,pred,correct_pred,train_op,cost,cost_c],
                                                                              feed_dict={
                                                                              l_encoder_embed_input:sources_batch,
                                                                              l_y:batch_y,training:True,
                                                                              u_encoder_embed_input:un_sources_batch,
                                                                              it_learning_rate: initial_learning_rate,
                                                                              keep_prob: 0.5,target_sequence_length: pad_source_lengths,
                                                                              un_target_sequence_length:un_pad_source_lengths,alpha:alpha_u})
                #computing for P R F1
                print "un", un_pred_batch
                for i in range(len(pred_batch)):
                    value=pred_batch[i]
                    top1=np.argpartition(a=-value,kth=0)[:1]
                    TRAIN_DIC.get(top1[0])[1] += 1
                    if c_pred[i]==True:
                        acc+=1
                        TRAIN_DIC.get(user_mask_id[i])[0]+=1 #REAL value a
                if(step%100==0 and step is not 0):
                    print ('TRAIN LOSS', train_cost,'Classifier Cost', classifier_cost)
                    print ("\n")
                loss=np.mean(batch_cost)
                cbatch_cost=np.mean(c_cost)
                classifier_cost+=cbatch_cost
                train_cost+=loss
                train_cost_list.append(train_cost)
                step+=1 # while
                count+=1
            #out of bacth
            #print "un",un_pred_batch

            # Precision Recall, F1
            P = []
            R = []
            for i in TRAIN_DIC.keys():
                if TRAIN_DIC.get(i)[1] == 0:
                    TRAIN_DIC.get(i)[1] = 1
                if TRAIN_DIC.get(i)[2] == 0:
                    TRAIN_DIC.get(i)[2] = 1
                Pi = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[1]
                Ri = TRAIN_DIC.get(i)[0] / TRAIN_DIC.get(i)[2]
                P.append(Pi)
                R.append(Ri)
            macro_R = np.mean(R)
            macro_P = np.mean(P)
            macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
            TRAIN_P.append(macro_P)
            TRAIN_R.append(macro_R)
            TRAIN_F1.append(macro_F1)
            TRAIN_ACC1.append(acc / (step * batch_size))
            print ('\nTRAIN RESULT')
            print ('macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1)
            print ('total train number', step * batch_size, 'learning rate', initial_learning_rate)
            print ('iter', epoch, 'Accuracy', acc / (step * batch_size),'TRAIN LOSS', train_cost)
            print ('\nepoch TEST')
            TEST_p, TEST_r, TEST_f1, TEST_acc1,  = test_model(sess, new_testS, new_testL, epoch)
            TEST_P.append(TEST_p)
            TEST_R.append(TEST_r)
            TEST_F1.append(TEST_f1)
            TEST_ACC1.append(TEST_acc1)
            Learning_rate.append(initial_learning_rate)

        #saver.save(sess, './temp/bk_tul_lvae.pkt')
        save_metrics(Learning_rate, TEST_P, TEST_R, TEST_F1, TEST_ACC1, root='./out_data/bk_vae_ltests.txt')
        save_metrics(Learning_rate, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1,root='./out_data/bk_vae_ltrains.txt')
        draw_pic_metric(TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, name='train')
        draw_pic_metric(TEST_P, TEST_R, TEST_F1, TEST_ACC1, name='test')



def test_model(sess,testS,testL,epoch):
    step = 0
    acc = 0
    testL = testL[:2500]
    testS = testS[:2500]

    tempU = list(set(label_list))
    TEST_DIC = {}
    for i in range(len(tempU)):
        TEST_DIC[i] = [0, 0, 0]  # use mask
    while step < len(testS) // batch_size:  #
        start_i = step * batch_size
        input_x = testS[start_i:start_i + batch_size]
        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        pad_source_lengths = []
        user_mask_id = []
        for source in input_x:
            pad_source_lengths.append(len(source)) #+1
        batch_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step = get_onehot(get_mask_index(testL[y_i], label_list))
            user_mask_id.append(get_mask_index(testL[y_i], label_list))
            TEST_DIC.get(get_mask_index(testL[y_i], label_list))[2] += 1  # Groud value Groud Truth a+c
            batch_y.append(xsy_step)
        f_pred,c_pred,pred_batch=sess.run([correct_pred_f,correct_pred,pred],feed_dict={l_encoder_embed_input:sources_batch,l_y: batch_y,
                                                                           keep_prob: 1.0,training:False,
                                                                           target_sequence_length: pad_source_lengths})
        for i in range(len(pred_batch)):
            value = pred_batch[i]
            top1 = np.argpartition(a=-value, kth=0)[:1]
            TEST_DIC.get(top1[0])[1] += 1
            if c_pred[i] == True:
                acc += 1
                TEST_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
        step+=1 # while
    # c_y = np.concatenate((np.array(pred_batch), np.array(batch_y)), axis=1)
    # for i, j in enumerate(c_y.tolist()):
    #     j.append(int(f_pred[i]))
    #     print j, i
    # Precision Recall, F1
    P = []
    R = []
    for i in TEST_DIC.keys():
        if TEST_DIC.get(i)[1] == 0:
            TEST_DIC.get(i)[1] = 1
        if TEST_DIC.get(i)[2] == 0:
            TEST_DIC.get(i)[2] = 1
        Pi = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[1]
        Ri = TEST_DIC.get(i)[0] / TEST_DIC.get(i)[2]
        P.append(Pi)
        R.append(Ri)
    macro_R = np.mean(R)
    macro_P = np.mean(P)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    print ('macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1)
    print ('iter', epoch, 'Accuracy For TEST', acc / (step * batch_size), 'total test number', step * batch_size)
    print "\n"
    return macro_P, macro_R, macro_F1, acc / (step * batch_size)

if __name__ == "__main__":
    train_model()
    print ('------------Model END------------')