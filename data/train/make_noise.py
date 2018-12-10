import skimage.util
from glob import glob
import scipy
import os
path = '/home/zhangshuhao/DenoiseGAN/data/test/DIV_clean'
noise_path = '/home/zhangshuhao/DenoiseGAN/data/test/DIV_noise'
data = glob(os.path.join(path, '*.*'))
count=0
for one in data:
    img = scipy.misc.imread(one, mode='RGB')
    name=one.split('/')[-1]
    img_noise = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True)
    scipy.misc.imsave(noise_path+'/'+name, img_noise)
    count += 1
    print(count)

