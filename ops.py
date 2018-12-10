#coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def res_block(input_x, out_channels=64, k=3, s=1, scope='res_block'):
    with tf.variable_scope(scope) as scope:
        x = input_x
        input_x = slim.conv2d_transpose(input_x, out_channels, k, s)
        input_x = slim.batch_norm(input_x, scope='bn1')
        input_x = tf.nn.relu(input_x)
        input_x = slim.conv2d_transpose(input_x, out_channels, k, s)
        input_x = slim.batch_norm(input_x, scope='bn2')
    
    return x+input_x
    
def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, [0,1,2,4,3])
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)      # tf.squeeze会从tensor中删除所有大小是1的维度
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    #tf.split(value, num_or_size_splits, axis, num=None, name='split')是把一个张量划分成几个子张量
    # value:准备切分的张量; num_or_size_splits:准备切分成几份; axis准备在第几个维度上切割
    # x的形状是(8, 64, 64, 256) tf.split会在第三维度也就是256将x切分为64份
    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

def down_sample_layer(input_x):
    K = 1
    #K = 4
    arr = np.zeros([K, K, 3, 3])
    arr[:, :, 0, 0] = 1.0 / K ** 2
    arr[:, :, 1, 1] = 1.0 / K ** 2
    arr[:, :, 2, 2] = 1.0 / K ** 2
    weight = tf.constant(arr, dtype=tf.float32)
    # tf.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    # input需要做卷积的输入图像，要求是一个tensor，具有[batch, in_height, in_width, in_channels]，要求类型是float32和float64之一
    # filter相当于cnn中的卷积核，它要求是一个tensor,具有[filter_height, filter_width, in_channels, out_channels]，具体含义是
    #[卷积核的高度，卷积核的宽度，图像通道数，卷积核的个数]，要求类型与参数input相同，第三维in_channels就是参数input的第四维。
    #strides,卷积时在每一维的步长，是一个一维向量，长度为4
    downscaled = tf.nn.conv2d(
        input_x, weight, strides=[1, K, K, 1], padding='SAME')
    #padding有SAME和VALID两种，当padding='VALID'时，new_height=new_width=[(W-F+1)/S]
    #当padding=‘SAME'时，new_height=new_width=[W/S]
    #W为输入的size, F为Filter的size，S为步长，[]为向上取整
    return downscaled

def leaky_relu(input_x, negative_slop=0.2):
    return tf.maximum(negative_slop*input_x, input_x)

def PSNR(real, fake):
    mse = tf.reduce_mean(tf.square(127.5*(real-fake)+127.5),axis=(-3,-2,-1))
    psnr = tf.reduce_mean(10 * (tf.log(255*255 / tf.sqrt(mse)) / np.log(10)))
    return psnr
