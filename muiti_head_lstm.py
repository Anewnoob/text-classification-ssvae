# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
from Imdb.utils.dump_data import dump_data
import os
import pickle  as pkl
from Imdb.utils.helper import *
from Imdb.utils.multihead import *

batch_size = 16
iter_num = 70  # iter_number
embedding_size = 512  # embedding size
c_hidden = 512  # classifer embedding
label_size = 2
Label_Train_Size = 2500
Unlabel_Train_Size = 30000
Test_Size = 2500
un_batch_size = int((Unlabel_Train_Size / Label_Train_Size) * batch_size)  # 250
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

dic_embeddings = tf.constant(dic_embeddings)

#-----------------------------------VAE-LOST------------------------------------
def get_onehot(index):
    x = [0] * label_size
    x[index] = 1
    return x

def embedding(features,is_reuse=None):
    """Customized function to transform batched x into embeddings."""
    # Convert indexes of words into embeddings.
    with tf.variable_scope('x_embed', reuse=is_reuse):
        W = tf.get_variable('W', initializer=dic_embeddings, trainable=True,dtype=tf.float32)
        print("initialize word embedding finished")
    word_vectors = tf.nn.embedding_lookup(W, features)
    return word_vectors

def classifer(encoder_embed_input,keep_prob=0.5,reuse=False,training = False):
    #encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_embed_input)
    encoder_input = embedding(encoder_embed_input)
    ma = multihead_attention(queries=encoder_input,keys=encoder_input,dropout_rate=keep_prob,is_training=training)
    outputs = feedforward(ma,  [c_hidden,embedding_size])
    o1 = tf.identity(outputs)
    for i in range(5):
        ma = multihead_attention(queries=o1, keys=o1,reuse = True)
        outputs = feedforward(ma, [c_hidden,embedding_size],reuse = True)
        o1 = tf.identity(outputs)

    outputs = tf.reshape(o1, [-1, 512*embedding_size])

    pred = tf.layers.dense(outputs, units=label_size)
    return pred

def get_cost_c(pred): #compute classifier cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=l_y))
    return cost
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

#tensor definition
keep_prob = tf.placeholder("float")
input_x = tf.placeholder(dtype=tf.int32)
training = tf.placeholder(tf.bool)
it_learning_rate = tf.placeholder("float")
l_y=tf.placeholder(dtype=tf.float32,shape=[batch_size,label_size])
target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
l_encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])


pred=classifer(l_encoder_embed_input,keep_prob =keep_prob,training = training)
cost_c=get_cost_c(pred)

cost = cost_c


pred = tf.nn.softmax(pred)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(l_y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

loss_to_minimize = cost
tvars = tf.trainable_variables()
gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=it_learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step,
                                               name='train_step')




def train_model():
    saver=tf.train.Saver()
    initial = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(initial)
        #saver.restore(sess, './temp/bk_tul_lvae.pkt')
        print('Read train & test data')
        initial_learning_rate = 0.0004
        learning_rate_len = 0.000008
        min_kl=0.0
        min_kl_epoch=min_kl #退火参数
        kl_lens = 0.008

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
            classifier_cost=0
            while step < len(new_trainS) // batch_size:
                start_i = step * batch_size
                input_x = new_trainS[start_i:start_i + batch_size]
                sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'],max_len = 512)
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

                pred_batch,c_pred,op,batch_cost,c_cost=sess.run([pred,correct_pred,train_op,cost,cost_c],
                                                                              feed_dict={
                                                                              l_encoder_embed_input:sources_batch,
                                                                              l_y:batch_y,training:True,
                                                                              it_learning_rate: initial_learning_rate,
                                                                              keep_prob: 0.5,target_sequence_length: pad_source_lengths
                                                                              })
                #computing for P R F1
                for i in range(len(pred_batch)):
                    value=pred_batch[i]
                    top1=np.argpartition(a=-value,kth=0)[:1]
                    TRAIN_DIC.get(top1[0])[1] += 1
                    if c_pred[i]==True:
                        acc+=1
                        TRAIN_DIC.get(user_mask_id[i])[0]+=1 #REAL value a
                loss=np.mean(batch_cost)
                cbatch_cost=np.mean(c_cost) #*alpha_epoch

                classifier_cost+=cbatch_cost
                train_cost+=loss
                train_cost_list.append(train_cost)

                step+=1 # while
                count+=1
            #out of bacth
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


        saver.save(sess, './temp/bk_tul_lvae.pkt')
        save_metrics(Learning_rate, TEST_P, TEST_R, TEST_F1, TEST_ACC1, root='./out_data/bk_vae_ltests.txt')
        save_metrics(Learning_rate, TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1,root='./out_data/bk_vae_ltrains.txt')
        draw_pic_metric(TRAIN_P, TRAIN_R, TRAIN_F1, TRAIN_ACC1, name='train')
        draw_pic_metric(TEST_P, TEST_R, TEST_F1, TEST_ACC1, name='test')

                #metric_compute(correct_pred)
def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch] #
def pad_sentence_batch(sentence_batch, pad_int,max_len = 384):
    new_sentence = []
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    for sentence in sentence_batch:
        if len(sentence)<max_len:
            sentence = sentence+[pad_int]*(max_len - len(sentence))
            new_sentence.append(sentence)
        else:
            sentence = sentence[:max_len]
            new_sentence.append(sentence)
    return new_sentence
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
def test_model(sess,testS,testL,epoch):
    step = 0
    acc = 0
    testS = testS[:2500]
    testL = testL[:2500]
    tempU = list(set(label_list))
    TEST_DIC = {}
    for i in range(len(tempU)):
        TEST_DIC[i] = [0, 0, 0]  # use mask
    while step < len(testS) // batch_size:  #
        start_i = step * batch_size
        input_x = testS[start_i:start_i + batch_size]

        sources_batch = pad_sentence_batch(input_x, vocab_to_int['<PAD>'],max_len=512)

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
        c_pred,pred_batch=sess.run([correct_pred,pred],feed_dict={l_encoder_embed_input:sources_batch, l_y: batch_y,
                                                                  training:False,
                                                                           keep_prob: 1.0,
                                                                           target_sequence_length: pad_source_lengths})
        for i in range(len(pred_batch)):
            value = pred_batch[i]
            #print value
            top1 = np.argpartition(a=-value, kth=0)[:1]
            TEST_DIC.get(top1[0])[1] += 1  # recommend value a+b

            if c_pred[i] == True:
                acc += 1
                TEST_DIC.get(user_mask_id[i])[0] += 1  # REAL value a
        step+=1 # while
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
    print ("\n")
    return macro_P, macro_R, macro_F1, acc / (step * batch_size)
def save_metrics(LEARN_RATE,TRAIN_P,TRAIN_R,TRAIN_F1,TRAIN_ACC1,root='result/bk_metric_vae.txt'):
    files=open(root,'a+')
    files.write('epoch \t learning_rate \t Precision \t Recall \t F1 \t ACC1 \n')
    for i in range(len(TRAIN_P)):
        files.write(str(i)+'\t')
        files.write(str(LEARN_RATE[i])+'\t')
        files.write(str(TRAIN_P[i]) + '\t'+str(TRAIN_R[i])+'\t'+str(TRAIN_F1[i])+'\t'+str(TRAIN_ACC1[i])+'\t'+'\n')
    files.close()
def draw_pic_metric(P,R,F1,ACC1,name='train'):
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

if __name__ == "__main__":
    train_model()
    print ('------------Model END------------')