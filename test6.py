# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from Imdb.utils.dump_data import dump_data
import clstm
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import os
import pickle as pkl
from Imdb.utils.helper import *

batch_size = 40
iter_num = 60  # iter_number
embedding_size = 250  # embedding size
vae_hidden = 512  # vae embeddings
c_hidden = 300  # classifer embedding
z_size = 100
label_size = 2
Label_Train_Size = 2500      #3047
Unlabel_Train_Size = 30000   #60610
Test_Size = 2500
data_path = "../data/aclImdb"


# -----------------data procession function------------------------
target_pkl_path = "../data/aclImdb/imdb_%d_%d_320.pkl" % (Label_Train_Size , embedding_size)
if not os.path.exists(target_pkl_path):
    """ start data processing"""
    dic_embeddings, new_unlabelS, new_testS, new_trainS, int_to_vocab, vocab_to_int, new_trainL, new_testL, label_list = dump_data(data_path,embedding_size,Label_Train_Size)
else:
    #key ='dic_embeddings','new_unlabelS','new_testS','new_trainS','int_to_vocab', 'vocab_to_int', 'train_label','test_label', 'label_list'
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
    print ("load data pkl successful!")

print (len(new_trainS),len(new_trainL))
print (len(new_testS),len(new_testL))
print (len(new_unlabelS))


un_batch_size = int((Unlabel_Train_Size / len(new_trainS) * batch_size))
new_unlabelS = new_unlabelS[:Unlabel_Train_Size]
new_testS = new_testS[:Test_Size]
new_testL = new_testL[:Test_Size]

class_name = ['negative','positive']
W_class_emb = load_class_embedding(vocab_to_int,class_name,dic_embeddings)
dic_embeddings=tf.constant(dic_embeddings)
dic_embeddings = tf.to_float(dic_embeddings,name = "dictofloat")
W_class_emb = tf.constant(W_class_emb)
W_class_emb = tf.to_float(W_class_emb,name = "classtofloat")
W_class = tf.get_variable('W_class', initializer=W_class_emb, trainable=True,dtype=tf.float32)
W_dic = tf.get_variable('W', initializer=dic_embeddings, trainable=True,dtype=tf.float32)
vae_dic = tf.get_variable('vae_dic', initializer=dic_embeddings, trainable=True, dtype=tf.float32)

# -----------------------------------SS-VAE MOLDE------------------------------------
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x

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

def embedding_class(l_y):
    y_pos = tf.argmax(l_y, -1)
    features = tf.to_int32(y_pos)
    label_vectors = tf.nn.embedding_lookup(W_class, features)
    return W_class,label_vectors


def classifier(encoder_embed_input, target_sequence_length,max_target_sequence_length, keep_prob=0.5, reuse=False):
    with tf.variable_scope("classifier", reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(c_hidden, forget_bias=1.0, state_is_tuple=True)
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)
        bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(c_hidden, forget_bias=1.0, state_is_tuple=True)
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_, dtype=tf.float32,
                                                            time_major=True)
        # attention
        fw_outputs, bw_outputs = outputs
        W = tf.Variable(tf.random_normal([c_hidden + label_size], stddev=0.1))
        h_state = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        h_state = tf.transpose(h_state, [1, 0, 2])
        M = tf.tanh(h_state)  # M = tanh(h_state)  (batch_size, seq_len, HIDDEN_SIZE)

        # joint embedding y
        W_class_tran = tf.transpose(W_class, [1, 0])  # e * c
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        masks = tf.expand_dims(masks, axis=-1)  # b * s * 1
        encoder_input_0 = tf.multiply(encoder_input, masks)
        x_emb_norm = tf.nn.l2_normalize(encoder_input_0, axis=2)  # b * s * e
        W_class_norm = tf.nn.l2_normalize(W_class_tran, axis=0)  # e * c  (250,2)
        G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c

        # contanct M,G
        M = tf.concat([M, G], 2)
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, c_hidden + label_size]), tf.reshape(W, [-1, 1])),
                                         (-1, max_target_sequence_length)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(h_state, [0, 2, 1]),  # (batch,hidden_size,seq_len)
                      tf.reshape(alpha, [-1, max_target_sequence_length, 1]))
        r = tf.squeeze(r)
        l_a = tf.tanh(r)  # (batch , HIDDEN_SIZE)
        l_c = tf.nn.dropout(l_a, keep_prob)
        l_c = layer_normalization(l_c)

        fw_states, bw_states = states
        l_fb = tf.concat([fw_states[0][1], bw_states[0][1]], 1)

        # dense
        FC_W_fb = tf.Variable(tf.truncated_normal([2 * c_hidden, c_hidden], stddev=0.1))
        tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W_fb))
        FC_b_fb = tf.Variable(tf.constant(0., shape=[c_hidden]))
        l_fb = tf.nn.xw_plus_b(l_fb, FC_W_fb, FC_b_fb)

        l_cfb = tf.concat([l_c, l_fb], 1)

        FC_W = tf.Variable(tf.truncated_normal([2 * c_hidden, label_size], stddev=0.1))
        tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W))
        FC_b = tf.Variable(tf.constant(0., shape=[label_size]))
        logits = tf.nn.xw_plus_b(l_cfb, FC_W, FC_b)

        FC_W_CLASS = tf.Variable(tf.truncated_normal([embedding_size, label_size], stddev=0.1))
        tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W_CLASS))
        FC_b_CLASS = tf.Variable(tf.constant(0., shape=[label_size]))
        logits_class = tf.nn.xw_plus_b(W_class, FC_W_CLASS, FC_b_CLASS)
        return logits,logits_class,l_c


def encoder(encoder_embed_input, l_y, logits,l_a,keep_prob=0.5, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        encode_lstm = tf.contrib.rnn.LSTMCell(vae_hidden, forget_bias=1.0, state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        # new_states = states[-1]  states  tuple  (c,h)

        #_, emb_y = embedding_class(y)  # (b,c)
        new_states = tf.concat([states[-1],l_a,l_y], 1)

        o_mean = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                   scope="z_mean")
        o_stddev = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                     scope="z_std")
        return states, o_mean, o_stddev


def decoder(decoder_embed_input,target_length, max_target_length, encode_state, batch_size, keep_prob,
            reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # decode_lstm = clstm.BasicLSTMCell(vae_hidden, label_size=label_size, embedding_size=embedding_size,
        #                                   forget_bias=1.0, state_is_tuple=True)
        decode_lstm = tf.contrib.rnn.LSTMCell(vae_hidden, forget_bias=1.0, state_is_tuple=True)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        decoder_initial_state = encode_state
        output_layer = Dense(embedding_size)  # TOTAL_SIZE

        decoder_embed_input = tf.strided_slice(decoder_embed_input,[0,0],[batch_size,-1],[1,1]) #del eos
        decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],1)
        decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
        #decoder_input = tf.concat([decoder_input,decoder_y], 2)  # dic_embedding+y(one-hot)+G
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
        masks = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32,
                                 name='masks')  # (batch_size,max_target_length)
        return output, predicting_logits, training_logits, masks


#------------------------------------Get loss------------------------------------------
def get_cost_c(logits,logits_class):  # compute classifier cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=l_y)) + 0.5*tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_y, logits=logits_class))
    return cost


def get_cost_l(encoder_embed_input, decoder_embed_input, l_y, target_sequence_length,
               max_target_sequence_length, batch_size,logits,l_a,reuse=False):
    encode_states, z_mean, z_stddev = encoder(encoder_embed_input, l_y,logits,l_a, keep_prob, reuse)
    samples = tf.random_normal(tf.shape(z_stddev))
    z = z_mean + tf.exp(z_stddev * 0.5) * samples
    # KL term-------------
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)

    #_, emb_y = embedding_class(l_y)  #(b,c)
    z_state = tf.concat([l_y,z],1)  #(64,100+300)
    # z_feature = tf.matmul(z_state, weights_de['w_1']) + biases_de['b_1']  #(64,300)
    # feature_loss =

    c_state = tf.nn.tanh(tf.matmul(z_state, weights_de['w_']) + biases_de['b_'])  #(64*64)
    #decoder_initial_state = clstm.LSTMStateTuple(c_state, encode_states[1])
    decoder_initial_state = LSTMStateTuple(c_state, encode_states[1])
    decoder_output, predicting_logits, training_logits, masks = decoder(decoder_embed_input,
                                                                                target_sequence_length,
                                                                                max_target_sequence_length,
                                                                                decoder_initial_state, batch_size,
                                                                                keep_prob,reuse)

    decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_embed_input)
    s_loss = tf.square(training_logits - decoder_input)  # batch,len,embeding_size
    mask_loss = tf.reduce_sum(tf.transpose(s_loss, [2, 0, 1]), 0)  # mask_loss (bacth_size,max_len_seq)
    encropy_loss = tf.reduce_mean(tf.multiply(mask_loss, masks), 1)  # 还原句子长度 其余位置都是0　　multiply　点乘
    cost = tf.add(encropy_loss, (latentscale_iter * (latent_loss)))  # cost  (batch_size)
    return cost


def get_cost_u(u_encoder_embed_input, u_decoder_embed_input):
    logits,logits_class,un_l_a = classifier(u_encoder_embed_input,un_target_sequence_length-1,un_max_target_sequence_length-1, keep_prob=keep_prob, reuse=True)
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=class_y, logits=logits_class))

    prob_y = tf.nn.softmax(logits)

    u_cost_l = get_cost_l(u_encoder_embed_input, u_decoder_embed_input, prob_y,
                            un_target_sequence_length, un_max_target_sequence_length, un_batch_size,logits,un_l_a,reuse=True)

    # un_pred_y = tf.argmax(prob_y, 1)
    # pred_y = tf.one_hot(un_pred_y, label_size)
    #loss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=prob_y, logits=prob_y))
    #cross_entropy = tf.reduce_mean(prob_y * tf.log(tf.clip_by_value(prob_y, 1e-10, 1.0)))

    un_label_cost = tf.reduce_mean(u_cost_l) + class_loss #- cross_entropy
    return un_label_cost, prob_y

def creat_y_scopus(label_y, seq_length):  # label data
    lcon_y = [label_y for j in range(seq_length)]
    return lcon_y

def creat_u_y_scopus(seq_length):  # unlabel data
    ucon_y = []
    for i in range(label_size):
        label_y = get_onehot(i)
        temp = []
        for j in range(un_batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        ucon_y.append(temp)
    return ucon_y


def creat_all_y_scopus(seq_length):  # unlabel data
    all_con_y = []
    for i in range(label_size):
        label_y = get_onehot(i)
        temp = []
        for j in range(batch_size):
            temp.append(creat_y_scopus(label_y, seq_length))
        all_con_y.append(temp)
    return all_con_y


#define the weight and bias dictionary
with tf.name_scope("weight_inital"):
    weights_de = {
        'w_': tf.Variable(tf.random_normal([z_size+label_size, vae_hidden], mean=0.0, stddev=0.01)),
        'w_1': tf.Variable(tf.random_normal([z_size +label_size, c_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([2 * c_hidden, label_size]))
    }
    biases_de = {
        'b_': tf.Variable(tf.random_normal([vae_hidden], mean=0.0, stddev=0.01)),
        'b_1': tf.Variable(tf.random_normal([c_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([label_size]))
    }

# tensor definition
keep_prob = tf.placeholder("float")
training = tf.placeholder(tf.bool)
alpha = tf.placeholder("float")
c_alpha = tf.placeholder("float")
it_learning_rate = tf.placeholder("float")
input_x = tf.placeholder(dtype=tf.int32)
l_y = tf.placeholder(dtype=tf.float32, shape=[batch_size, label_size])
vae_y = tf.placeholder("float", [batch_size, None, label_size])  # vae_y
vae_y_all = tf.placeholder("float", [label_size, batch_size, None, label_size])
vae_y_u = tf.placeholder("float", [label_size, un_batch_size, None, label_size])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
u_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None])
u_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[un_batch_size, None])
latentscale_iter = tf.placeholder(dtype=tf.float32)
class_y = tf.constant(name='class_y', shape=[label_size, label_size], dtype=tf.float32,
                      value=np.identity(label_size), )



logits,logits_class,l_a = classifier(l_encoder_embed_input, target_sequence_length-1,max_target_sequence_length-1, keep_prob=keep_prob)
cost_c = get_cost_c(logits,logits_class)
tf.add_to_collection("losses_c", cost_c)
cost_c = tf.add_n(tf.get_collection("losses_c"))

cost_l = get_cost_l(l_encoder_embed_input, l_decoder_embed_input, l_y,target_sequence_length, max_target_sequence_length,
                    batch_size,logits,l_a)

cost_u, prob_y = get_cost_u(u_encoder_embed_input, u_decoder_embed_input)
cost = c_alpha*cost_c + cost_l + alpha*cost_u
#cost = cost_c + 0.5*tf.reduce_mean(cost_l) + cost_u

pred = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(l_y, 1))
correct_pred_f = tf.cast(correct_pred, tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)


#-----------------------------------------main---------------------------------------------
def train_model():
    saver = tf.train.Saver()
    initial = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as  sess:
        sess.run(initial)
        #saver.restore(sess, './temp/imdb_new_version_8_dic_nosclstm_2500.pkt')
        print('Read train & test data')
        initial_learning_rate = 0.0005
        learning_rate_len = 0.000004
        min_kl = 0.0
        min_kl_epoch = min_kl  # 退火参数
        kl_lens = 0.002

        # -----------------------------------
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
        alpha_epoch = 1
        alpha_value = (2.0 - 1.0) / iter_num
        train_cost_list = []
        T1 = 10
        T2 = 40
        af = 0.25

        alpha_u = 0.
        alpha_c = 1.
        for epoch in range(iter_num):
            initial_learning_rate -= learning_rate_len
            if (initial_learning_rate <= 0):
                initial_learning_rate = 0.000001
            step = 0
            acc = 0
            train_cost = 0
            label_cost = 0
            unlabel_cost = 0
            classifier_cost = 0
            if epoch > T1:
                alpha_u = (epoch-T1) / (T2-T1) * af
                if epoch > T2:
                    alpha_u = af

            while step < len(new_trainS) // batch_size:  # 2500/25
                start_i = step * batch_size
                input_x = new_trainS[start_i:start_i + batch_size]
                un_start_i = step * un_batch_size
                input_ux = new_unlabelS[un_start_i:un_start_i + un_batch_size]

                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
                encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])  # append <EOS> to seq
                input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])  # padding

                un_sources_batch = pad_sentence_batch(input_ux, vocab_to_int['<PAD>'])
                un_encode_batch = eos_sentence_batch(input_ux, vocab_to_int['<EOS>'])
                un_input_batch = pad_sentence_batch(un_encode_batch, vocab_to_int['<PAD>'])

                # unlabel
                un_pad_source_lengths = []
                for source in input_ux:
                    un_pad_source_lengths.append(len(source)+1)  # +1 is add <GO> word lenth in decoder
                # label
                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source)+1)  # +1

                target_maxlength = len(input_batch[0])   # get max length  +1
                un_target_maxlength = len(un_input_batch[0])   # get max length  +1

                if min_kl_epoch < 1.0:
                    min_kl_epoch = min_kl + count * kl_lens
                else:
                    min_kl_epoch = 1.0
                batch_y = []
                decode_y = []
                user_mask_id = []
                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainL[y_i], label_list))
                    user_mask_id.append(get_mask_index(new_trainL[y_i], label_list))
                    TRAIN_DIC.get(get_mask_index(new_trainL[y_i], label_list))[2] += 1  # Groud value Groud Truth a+c
                    batch_y.append(xsy_step)

                # acc_un, ulab, p_y,acc_l, wrong_l_cost,op,c_pred,batch_cost,c_cost,u_cost
                p_y,op, pred_batch, c_pred, batch_cost, l_cost, c_cost,u_cost = sess.run(
                    [prob_y,optimizer, pred, correct_pred, cost, cost_l, cost_c,cost_u
                     ],  # un_acc,L_ulab,prob_y,cost_c,cost_u,pred,correct_pred,
                    feed_dict={
                               l_encoder_embed_input: sources_batch, l_decoder_embed_input: input_batch,
                               l_y: batch_y,
                               u_decoder_embed_input: un_input_batch, u_encoder_embed_input: un_sources_batch,
                               it_learning_rate: initial_learning_rate, latentscale_iter: min_kl_epoch,
                               keep_prob: 0.5, target_sequence_length: pad_source_lengths,
                               un_target_sequence_length: un_pad_source_lengths, alpha: alpha_u, c_alpha: alpha_c
                               })

                for i in range(len(pred_batch)):
                    value = pred_batch[i]
                    top1 = np.argpartition(a=-value, kth=0)[:1]
                    TRAIN_DIC.get(top1[0])[1] += 1
                    if c_pred[i] == True:
                        acc += 1
                        TRAIN_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
                # print logit.shape
                if (step % 50 == 0 and step is not 0):
                    print ('min_kl_epoch', min_kl_epoch)
                    print (
                    'TRAIN LOSS', train_cost, 'LABEL COST', label_cost, 'Unlabel Cost', unlabel_cost, 'Classifier Cost',
                    classifier_cost)
                    print ("\n")
                loss = np.mean(batch_cost)
                lbatch_cost = np.mean(l_cost)
                ubatch_cost = np.mean(u_cost)
                cbatch_cost = np.mean(c_cost)  # *alpha_epoch

                classifier_cost += cbatch_cost
                unlabel_cost += ubatch_cost
                label_cost += lbatch_cost
                train_cost += loss
                train_cost_list.append(train_cost)
                step += 1  # while
                count += 1
            #print ("acc_l",acc_l,"l_cost",l_cost,"lab",wrong_l_cost)#,"y",batch_y
            print ("p_y", p_y)
            alpha_epoch += alpha_value

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
            macro_R = np.mean(R,dtype = np.float32)
            macro_P = np.mean(P,dtype = np.float32)
            macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
            TRAIN_P.append(macro_P)
            TRAIN_R.append(macro_R)
            TRAIN_F1.append(macro_F1)
            TRAIN_ACC1.append(acc / (step * batch_size))
            print ('\nTRAIN RESULT')
            print ('macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1)
            print ('total train number', step * batch_size, 'learning rate', initial_learning_rate)
            print ('iter', epoch, 'Accuracy', acc / (step * batch_size), 'TRAIN LOSS', train_cost)
            print ('\nepoch TEST')
            TEST_p, TEST_r, TEST_f1, TEST_acc1, = test_model(sess, new_testS, new_testL, epoch)
            TEST_P.append(TEST_p)
            TEST_R.append(TEST_r)
            TEST_F1.append(TEST_f1)
            TEST_ACC1.append(TEST_acc1)
            Learning_rate.append(initial_learning_rate)
            # if epoch == 14:
            #     model_path = './temp/test5_dic_sclstm_%d.pkt' % Label_Train_Size
            #     saver.save(sess, model_path)
        save_metrics(Learning_rate, TEST_P, TEST_R, TEST_F1, TEST_ACC1, root='./out_data/bk_vae_ltests.txt')
        save_metrics(Learning_rate, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, root='./out_data/bk_vae_ltrains.txt')
        draw_pic_metric(TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, name='train')
        draw_pic_metric(TEST_P, TEST_R, TEST_F1, TEST_ACC1, name='test')


def test_model(sess, testS, testL, epoch):
    step = 0
    acc = 0
    tempU = list(set(label_list))
    TEST_DIC = {}
    for i in range(len(tempU)):
        TEST_DIC[i] = [0, 0, 0]  # use mask
    while step < len(testS) // batch_size:  #
        start_i = step * batch_size
        input_x = testS[start_i:start_i + batch_size]

        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])
        encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])
        input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'], max_len=800)

        pad_source_lengths = []
        user_mask_id = []
        for source in input_x:
            pad_source_lengths.append(len(source) + 1)  # +1
        batch_y = []
        for y_i in range(start_i, start_i + batch_size):
            xsy_step = get_onehot(get_mask_index(testL[y_i], label_list))
            user_mask_id.append(get_mask_index(testL[y_i], label_list))
            TEST_DIC.get(get_mask_index(testL[y_i], label_list))[2] += 1  # Groud value Groud Truth a+c
            batch_y.append(xsy_step)
        f_pred, c_pred, pred_batch = sess.run([correct_pred_f, correct_pred, pred],
                                              feed_dict={l_encoder_embed_input: sources_batch, l_y: batch_y,
                                                         keep_prob: 1.0, l_decoder_embed_input: input_batch,
                                                         target_sequence_length: pad_source_lengths})
        for i in range(len(pred_batch)):
            value = pred_batch[i]
            top1 = np.argpartition(a=-value, kth=0)[:1]
            TEST_DIC.get(top1[0])[1] += 1  # recommend value a+b

            if c_pred[i] == True:
                acc += 1
                TEST_DIC.get(user_mask_id[i])[0] += 1  # REAL value a

        step += 1  # while
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
    macro_R = np.mean(R,dtype = np.float32)
    macro_P = np.mean(P,dtype = np.float32)
    macro_F1 = 2 * macro_P * macro_R / (macro_P + macro_R)
    print ('macro-p', macro_P, 'macro-r', macro_R, 'macro-f1', macro_F1)
    print ('iter', epoch, 'Accuracy For TEST', acc / (step * batch_size), 'total test number', step * batch_size)
    print ("\n")
    return macro_P, macro_R, macro_F1, acc / (step * batch_size)


if __name__ == "__main__":
    train_model()
    print ('------------Model END------------')