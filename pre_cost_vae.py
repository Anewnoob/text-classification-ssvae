# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from utils.dump_data import dump_data
import clstm
import tensorflow.contrib.slim as slim
import os
import cPickle  as pkl
from utils.helper import *

batch_size = 56
iter_num = 30  # iter_number
embedding_size = 250  # embedding size
vae_hidden = 512  # vae embeddings
c_hidden = 512  # classifer embedding
z_size = 100
label_size = 2
Label_Train_Size = 2500
Test_Size = 2500
data_path = "../data/aclImdb"


# -----------------data procession function------------------------
target_pkl_path = "../data/aclImdb/imdb_%d_%d.pkl" % (Label_Train_Size , embedding_size)
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

class_name = ['negative','positive']
W_class_emb = load_class_embedding(vocab_to_int,class_name,dic_embeddings)
dic_embeddings=tf.constant(dic_embeddings)
dic_embeddings = tf.to_float(dic_embeddings,name = "dictofloat")
W_class_emb = tf.constant(W_class_emb)
W_class_emb = tf.to_float(W_class_emb,name = "classtofloat")

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

def embedding(features,is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope('x_embed', reuse=is_reuse):
        W = tf.get_variable('W', initializer=dic_embeddings, trainable=True,dtype=tf.float32)
        print("initialize word embedding finished")
    word_vectors = tf.nn.embedding_lookup(W, features)
    return word_vectors, W

def embedding_class(is_reuse=None):
    """Customized function to transform batched y into embeddings."""
    # Convert indexes of words into embeddings.

    with tf.variable_scope('class_embed', reuse=is_reuse):
        W_class = tf.get_variable('W_class', initializer=W_class_emb, trainable=True,dtype=tf.float32)
        print("initialize class embedding finished")
    return W_class

def classifer(encoder_embed_input, max_target_sequence_length, keep_prob=0.5, reuse=False):
    with tf.variable_scope("classifier", reuse=reuse):
        encoder_input, W_norm = embedding(encoder_embed_input,is_reuse=reuse)  # b * s * e
        encoder_input = tf.cast(encoder_input, tf.float32)
        #encoder_input = tf.nn.embedding_lookup(W_x, encoder_embed_input)
        #encoder_input = tf.cast(encoder_input, tf.float32)
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
        W = tf.Variable(tf.random_normal([c_hidden+label_size], stddev=0.1))
        h_state = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        h_state = tf.transpose(h_state, [1, 0, 2])
        M = tf.tanh(h_state)  # M = tanh(h_state)  (batch_size, seq_len, HIDDEN_SIZE)

        # joint embedding y
        W_class = embedding_class(is_reuse=reuse)  # b * e, c * e
        W_class = tf.cast(W_class, tf.float32)
        W_class_tran = tf.transpose(W_class, [1, 0])  # e * c

        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        masks = tf.expand_dims(masks, axis=-1)  # b * s * 1

        encoder_input_0 = tf.multiply(encoder_input, masks)

        x_emb_norm = tf.nn.l2_normalize(encoder_input_0, dim=2)  # b * s * e
        W_class_norm = tf.nn.l2_normalize(W_class_tran, dim=0)  # e * c
        G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c

        # contanct M,G
        M = tf.concat([M, G], 2)

        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, c_hidden+label_size]),
                                                       tf.reshape(W, [-1, 1])),
                                             (-1, max_target_sequence_length)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(h_state, [0, 2, 1]),  # (batch,hidden_size,seq_len)
                          tf.reshape(alpha, [-1, max_target_sequence_length, 1]))
        r = tf.squeeze(r)
        l_c = tf.tanh(r)  # (batch , HIDDEN_SIZE)
        l_c = tf.nn.dropout(l_c, keep_prob)
        l_c = layer_normalization(l_c)


        FC_W = tf.Variable(tf.truncated_normal([c_hidden, label_size], stddev=0.1))
        # L2
        tf.add_to_collection("losses_c", tf.contrib.layers.l2_regularizer(0.001)(FC_W))
        FC_b = tf.Variable(tf.constant(0., shape=[label_size]))
        pred = tf.nn.xw_plus_b(l_c, FC_W, FC_b)
        return pred

def encoder(encoder_embed_input, y, keep_prob=0.5, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
        input_ = tf.transpose(encoder_input, [1, 0, 2])
        encode_lstm = tf.contrib.rnn.LSTMCell(vae_hidden, forget_bias=1.0, state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_lstm, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)
        # new_states = states[-1]  states  tuple  (c,h)

        new_states = tf.concat([states[-1], y], 1)
        o_mean = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                   scope="z_mean")
        o_stddev = tf.contrib.layers.fully_connected(inputs=new_states, num_outputs=z_size, activation_fn=None,
                                                     scope="z_std")
        return outputs, states, o_mean, o_stddev


def decoder(decoder_embed_input, decoder_y, target_length, max_target_length, encode_state, batch_size, keep_prob,
            reuse=False):
    with tf.variable_scope("decoder", reuse=reuse):
        # decode_lstm = tf.contrib.rnn.LSTMCell(vae_hidden, forget_bias=1.0, state_is_tuple=True)
        decode_lstm = clstm.BasicLSTMCell(vae_hidden, label_size=label_size, embedding_size=embedding_size,
                                          forget_bias=1.0, state_is_tuple=True)
        decode_cell = tf.contrib.rnn.DropoutWrapper(decode_lstm, output_keep_prob=keep_prob)
        decoder_initial_state = encode_state
        output_layer = Dense(embedding_size)  # TOTAL_SIZE
        decoder_input_ = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), decoder_embed_input],
                                   1)  # add   1  GO to the end
        decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_input_)
        decoder_input = tf.concat([decoder_input, decoder_y], 2)  # dic_embedding+y(one-hot)
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
        # target = tf.concat([target_input, tf.fill([batch_size, 1], vocab_to_int['<EOS>'])], 1)  #
        target = decoder_embed_input
        return output, predicting_logits, training_logits, masks, target


#------------------------------------Get loss------------------------------------------
def get_cost_c(pred):  # compute classifier cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=l_y))
    return cost


def get_cost_l(encoder_embed_input, decoder_embed_input, l_y, decoder_y, target_sequence_length,
               max_target_sequence_length, batch_size, reuse=False):
    encode_outputs, encode_states, z_mean, z_stddev = encoder(encoder_embed_input, l_y, keep_prob, reuse)
    samples = tf.random_normal(tf.shape(z_stddev))
    z = z_mean + tf.exp(z_stddev * 0.5) * samples

    # KL term-------------
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
    c_state = tf.nn.softplus(tf.matmul(z, weights_de['w_']) + biases_de['b_'])
    #c_state = layer_normalization(c_state)

    decoder_initial_state = clstm.LSTMStateTuple(c_state, encode_states[1])
    decoder_output, predicting_logits, training_logits, masks, target = decoder(decoder_embed_input, decoder_y,
                                                                                target_sequence_length,
                                                                                max_target_sequence_length,
                                                                                decoder_initial_state, batch_size,
                                                                                keep_prob, reuse)

    decoder_input = tf.nn.embedding_lookup(dic_embeddings, decoder_embed_input)
    s_loss = tf.square(training_logits - decoder_input)  # batch,len,embeding_size
    mask_loss = tf.reduce_sum(tf.transpose(s_loss, [2, 0, 1]), 0)  # mask_loss (bacth_size,max_len_seq)
    encropy_loss = tf.reduce_mean(tf.multiply(mask_loss, masks), 1)  # 还原句子长度 其余位置都是0　　multiply　点乘
    cost = tf.add(encropy_loss, (latentscale_iter * (latent_loss)))  # cost  (batch_size)
    return cost


def get_cost_l_all(encoder_embed_input, decoder_embed_input, l_y, decoder_y, target_sequence_length,
                   max_target_sequence_length, batch_size, reuse=False):
    flag = 0
    real_label_cost = get_cost_l(encoder_embed_input, decoder_embed_input, l_y, decoder_y, target_sequence_length,
                                 max_target_sequence_length, batch_size, reuse=False)
    for label in range(label_size):
        y_i = get_onehot(label)
        batch_y = [y_i] * batch_size
        if batch_y != l_y:
            wrong_label_cost = get_cost_l(encoder_embed_input, decoder_embed_input, batch_y, vae_y_all[label],
                                          target_sequence_length,
                                          max_target_sequence_length, batch_size, reuse=True)
            wrong_cost = tf.expand_dims(wrong_label_cost, 1)
            # sub_cost = tf.abs(real_label_cost - wrong_label_cost)
            # sub_cost_expand = tf.expand_dims(sub_cost, 1)
            if flag == 0:
                wrong_cost_all = tf.identity(wrong_cost)
                #sub_cost_all = tf.identity(sub_cost_expand)
                flag = 1
            else:
                wrong_cost_all = tf.concat([wrong_cost_all, wrong_cost], 1)
                #sub_cost_all = tf.concat([sub_cost_all,sub_cost_expand],1)
    # vae_pred_index = tf.argmin(sub_cost_all, 1)
    # vae_y = tf.one_hot(vae_pred_index, label_size)
    # sub_cost_mean = tf.reduce_mean(sub_cost_all, 1)

    wrong_cost_mean = tf.reduce_mean(wrong_cost_all, 1)
    vae_pred_index = tf.argmin(wrong_cost_all, 1)
    vae_y = tf.one_hot(vae_pred_index, label_size)
    label_0 = tf.slice(wrong_cost_all, [0, 0], [-1, 1])
    label_1 = tf.slice(wrong_cost_all, [0, 1], [-1, 1])
    new_wrong_cost = tf.concat([label_1, label_0], 1)
    #vae_pred = tf.tanh(new_wrong_cost)
    vae_pred_y = tf.nn.softmax(new_wrong_cost)

    cost_vae = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vae_pred_y, labels=l_y))
    #vae_pred_y = tf.nn.softmax(vae_pred)
    same_pred = tf.equal(tf.argmax(vae_y, 1), tf.argmax(l_y, 1))
    accuracy = tf.reduce_mean(tf.cast(same_pred, tf.float32))

    target_cost = real_label_cost - 0.1*wrong_cost_mean + cost_vae
    return target_cost, real_label_cost, wrong_cost_all, accuracy

def creat_y_scopus(label_y, seq_length):  # label data
    lcon_y = [label_y for j in range(seq_length)]
    return lcon_y


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
        'w_': tf.Variable(tf.random_normal([z_size, vae_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([2 * c_hidden, label_size]))
    }
    biases_de = {
        'b_': tf.Variable(tf.random_normal([vae_hidden], mean=0.0, stddev=0.01)),
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

target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
un_target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
un_max_target_sequence_length = tf.reduce_max(un_target_sequence_length, name='max_target_len')
l_decoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
latentscale_iter = tf.placeholder(dtype=tf.float32)

pred = classifer(l_encoder_embed_input, max_target_sequence_length - 1, keep_prob=keep_prob)
cost_c = get_cost_c(pred)
tf.add_to_collection("losses_c", cost_c)
cost_c = tf.add_n(tf.get_collection("losses_c"))

target_cost_l, cost_l, wrong_cost_l, l_acc = get_cost_l_all(l_encoder_embed_input, l_decoder_embed_input, l_y, vae_y,
                                                            target_sequence_length, max_target_sequence_length,
                                                            batch_size)
cost = cost_c+tf.reduce_mean(target_cost_l) #+ tf.reduce_mean(cost_u)  # +alpha*un_cost_c


pred = tf.nn.softmax(pred)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(l_y, 1))
correct_pred_f = tf.cast(correct_pred, tf.int32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate).minimize(cost)


#-----------------------------------------main---------------------------------------------
def train_model():
    #exclude = ['classifier', 'classifier', 'ln', 'Variable_1', 'Adam_1']
    # exclude = ['classifier','classifier','bidirectional_rnn','bw','multi_rnn_cell','cell_0','basic_lstm_cell','kernel','Adam_1']
    #variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    saver = tf.train.Saver()
    initial = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as  sess:
        sess.run(initial)
        #saver.restore(sess, './temp/imdb_2500_pre_512_50_topic.pkt')
        print('Read train & test data')
        initial_learning_rate = 0.0004
        min_kl = 0.0
        min_kl_epoch = min_kl  # 退火参数
        kl_lens = 0.008
        # -----------------------------------
        count = 0
        for epoch in range(iter_num):
            if (initial_learning_rate <= 0):
                initial_learning_rate = 0.000001
            step = 0
            train_cost = 0
            label_cost = 0


            while step < len(new_trainS) // batch_size:  # 2500/25
                start_i = step * batch_size
                input_x = new_trainS[start_i:start_i + batch_size]


                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'])  # padding the same(max lenth)
                encode_batch = eos_sentence_batch(input_x, vocab_to_int['<EOS>'])  # append <EOS> to seq
                input_batch = pad_sentence_batch(encode_batch, vocab_to_int['<PAD>'])  # padding

                pad_source_lengths = []
                for source in input_x:
                    pad_source_lengths.append(len(source) + 1)  # +1

                target_maxlength = len(input_batch[0]) + 1  # get max length  +1


                if min_kl_epoch < 1.0:
                    min_kl_epoch = min_kl + count * kl_lens
                else:
                    min_kl_epoch = 1.0
                batch_y = []
                decode_y = []

                for y_i in range(start_i, start_i + batch_size):
                    xsy_step = get_onehot(get_mask_index(new_trainL[y_i], label_list))
                    decode_y.append(creat_y_scopus(xsy_step, target_maxlength))  # copy
                    batch_y.append(xsy_step)
                #decode_uy = creat_u_y_scopus(un_target_maxlength)
                decode_y_all = creat_all_y_scopus(target_maxlength)
                # acc_un, ulab, p_y,c_pred,batch_cost,c_cost,u_cost
                acc,acc_l, wrong_l_cost, op, batch_cost, l_cost = sess.run(
                    [accuracy,l_acc,wrong_cost_l,optimizer,cost,cost_l],  # un_acc,L_ulab,prob_y,cost_c,cost_u,pred,correct_pred,
                    feed_dict={vae_y: decode_y, vae_y_all: decode_y_all, #vae_y_u: decode_uy,
                               l_encoder_embed_input: sources_batch, l_decoder_embed_input: input_batch,
                               l_y: batch_y,
                               it_learning_rate: initial_learning_rate, latentscale_iter: min_kl_epoch,
                               keep_prob: 0.5, target_sequence_length: pad_source_lengths,
                               })

                # print logit.shape
                if (step % 50 == 0 and step is not 0):
                    print ('min_kl_epoch', min_kl_epoch)
                    print ('TRAIN LOSS', train_cost, 'LABEL COST', label_cost,'acc',acc)
                    print ("\n")
                loss = np.mean(batch_cost)
                lbatch_cost = np.mean(l_cost)
                label_cost += lbatch_cost
                train_cost += loss
                step += 1  # while
                count += 1
            print ("acc_l",acc_l)#"l_cost",l_cost,"lab",wrong_l_cost,"y",batch_y
            print ("lab",wrong_l_cost)
            print ('\nTRAIN RESULT')
            print ('total train number', step * batch_size, 'learning rate', initial_learning_rate)
            print ('iter', epoch, 'TRAIN LOSS', train_cost)
        model_path = './temp/imdb_new_pre_%d.pkt' % Label_Train_Size
        saver.save(sess, model_path)

if __name__ == "__main__":
    train_model()
    print ('------------Model END------------')