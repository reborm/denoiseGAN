import skimage.util
from glob import glob
import scipy
import os
clean_path = '/home/zhangshuhao/DenoiseGAN/data/train/Poly_clean'
noise_path = '/home/zhangshuhao/DenoiseGAN/data/train/Poly_noise'
clean_data = glob(os.path.join(clean_path, '*'))
noise_data = glob(os.path.join(noise_path, '*'))
count=0
for one in noise_data:
    new_name = one.replace('real', '')
    os.rename(one, new_name)
    count += 1
    print(count)
count_1=0
for one in clean_data:
    new_name = one.replace('mean', '')
    os.rename(one, new_name)
    count_1 += 1
    print(count_1)

