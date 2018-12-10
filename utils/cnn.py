# encoding=utf-8
import numpy as np
import tensorflow as tf

def cnn_conv_net(CNN_X,training = False) :
    conv_1 = tf.layers.conv2d(inputs=CNN_X,
                                filters=32,
                                kernel_size=[5,5],
                                padding='SAME',
                                activation=None)
    conv_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1,training = training))

    pool_1=tf.layers.max_pooling2d(inputs=conv_1,
                                   pool_size=[2,2],
                                   strides=2,
                                   padding='VALID')
    conv_2 = tf.layers.conv2d(  inputs=pool_1,
                                filters=64,
                                kernel_size=[3, 3],
                                padding="same",
                                activation=None)
    conv_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=training))

    pool_2 = tf.layers.max_pooling2d(inputs=conv_2,
                                     pool_size=[2, 2],
                                     strides=2,
                                     padding="valid")

    spp_bin_1 = tf.layers.conv2d(inputs=pool_2,  #####(?,75,1,32)
                                 filters=32,
                                 kernel_size=[5,5],
                                 padding='SAME',
                                 activation=tf.nn.relu)

    spp_bin_1 = tf.layers.max_pooling2d(inputs=spp_bin_1,  ##########(?,28,1,32)
                                        pool_size=[2, 2],
                                        strides=2,
                                        padding='VALID')

    spp_bin_1 = tf.layers.batch_normalization(spp_bin_1)
    spp_bin_1 = tf.layers.flatten(spp_bin_1)  ###(?,896)

    spp_bin_4 = tf.layers.conv2d(inputs=pool_2,  #####(?,75,1,32)
                                 filters=32,
                                 kernel_size=[3, 3],
                                 padding='SAME',
                                 activation=tf.nn.relu)

    spp_bin_4 = tf.layers.max_pooling2d(inputs=spp_bin_4,  ##########(?,23,1,32)
                                        pool_size=[2, 2],
                                        strides=2,
                                        padding='VALID')
    spp_bin_4 = tf.layers.batch_normalization(spp_bin_4)
    spp_bin_4 = tf.layers.flatten(spp_bin_4)  ####(?,736)


    spp_bin_8 = tf.layers.conv2d(inputs=pool_2,  #####(?,75,1,32)
                                 filters=32,
                                 kernel_size=[4, 4],
                                 padding='SAME',
                                 activation=tf.nn.relu)

    spp_bin_8 = tf.layers.max_pooling2d(inputs=spp_bin_8,  ##########(?,18,1,32)
                                        pool_size=[2, 2],
                                        strides=2,
                                        padding='VALID')
    spp_bin_8 = tf.layers.batch_normalization(spp_bin_8)
    spp_bin_8 = tf.layers.flatten(spp_bin_8)  ###(?,576)

    concat_1 = tf.concat([spp_bin_1, spp_bin_4, spp_bin_8], axis=1)  ###(?,2208)

    ful_in = tf.layers.flatten(concat_1)

    return ful_in