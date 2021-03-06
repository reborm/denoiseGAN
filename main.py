#coding=utf-8
import tensorflow as tf
import os
from denoiseGAN import *

flags = tf.app.flags
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.999, 'beta2')
flags.DEFINE_float('lambd', 0.001, 'coeff for adversarial loss')
flags.DEFINE_string('dataset_dir', 'data', 'dataset directory')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'checkpoint directory')
flags.DEFINE_string('sample_dir', 'sample', 'sample directory')
flags.DEFINE_string('test_dir', 'test', 'test directory')
flags.DEFINE_string('model_dir', 'DIV2K', 'using imagenet dataset')
flags.DEFINE_string('logs_dir', 'logs', 'log directory')
flags.DEFINE_bool('is_crop', False, 'crop images')
flags.DEFINE_integer('epoches', 100, 'training epoches')
flags.DEFINE_integer('fine_size', 256, 'fine size')
flags.DEFINE_string('train_set_clean', 'DIV2K_mean', 'clean train data set')
flags.DEFINE_string('train_set_noise', 'DIV2K_real', 'noise train dataset')
flags.DEFINE_string('val_set_clean', 'DIV2K_mean', 'clean val data set')
flags.DEFINE_string('val_set_noise', 'DIV2K_real', 'noise val data set')
#flags.DEFINE_string('test_set_clean', 'mean', 'clean test data set')
#flags.DEFINE_string('test_set_noise', 'real', 'noise test data set')
flags.DEFINE_string('test_dnd_noise', 'dnd_noise', 'the croped dnd noise images for test')
flags.DEFINE_bool('is_testing', True, 'training or testing')
flags.DEFINE_bool('is_training', False, 'training or testing')
FLAGS = flags.FLAGS

def check_dir():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.mkdir(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.logs_dir):
        os.mkdir(FLAGS.logs_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.mkdir(FLAGS.test_dir)


def main(_):
    check_dir()
    # per_process_gpu_memory_fraction设置占用gpu显存的比重
    # all_growth允许按需设置显存
    # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
    # log_device_placement=True:是否打印设备分配日志
    # allow_soft_placement=True:如果指定的设备不存在，允许tf自动分配设备
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        denoise_gan = denoiseGAN(FLAGS, batch_size=8, input_height=256, input_width=256, input_channels=3, sess=sess)
        denoise_gan.build_model()
        if FLAGS.is_training:
            denoise_gan.train()
        if FLAGS.is_testing:
            denoise_gan.test()
if __name__=='__main__':
    # tf.device指定运行设备,/cpu:0中的0表示设备号，tf不区分cpu的设备号,设置为0即可.gpu区分设备号\gpu:0和\gpu:1表示两张不同的显卡
    with tf.device('/gpu:0'):
        tf.app.run()    #tf.app.run()的作用是处理flag解析，然后执行main()函数，如果代码的入口函数不叫main而是一个其他名字的函数
                        #比如,test(),则入口应该写成tf.app.run(test())
