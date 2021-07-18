#!/usr/bin/env python3


import os
#os.system('pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl')


from glob import glob
#from network import compress
#from network import compress
from  test_cvpr_blocks_leaky_GLLMM_diectly_bits_github import compress
from  test_cvpr_blocks_leaky_GLLMM_diectly_bits_github import decompress
import numpy as np

#path = '/SSD/keda_24/'
path = '/SSD/keda_example/'
checkpoint_dir = '/SSD/Fhs_data/TF_GMM_hybrid/block_leaky_GLLMM_128_0.0075/model.ckpt-1500001'
root_dir = '/output'

save_image_name_path = '/home/fhs/test_image'
if not os.path.isdir(save_image_name_path):
  os.makedirs(save_image_name_path)
  
save_recImage_path = '/home/fhs/rec_image'
if not os.path.isdir(save_recImage_path):
  os.makedirs(save_recImage_path)

def endcoder_main():
  eval_bpp_list = []
  bpp_real_list = []
  time_list = []
  psnr_list = []
  msssim_list = []
  # save_image_name_path = checkpoint_dir.split('/')[-2]
  # print(save_image_name_path)

    
  for image_file in glob(path+'*.png'):
    print(image_file[:])
    image_name_path = image_file.split('/')[-1]
    input = image_file
    output = save_image_name_path+'/'+image_name_path + '.npz'
    print(input)
    print(output)
    #output = 'images/'
    num_filters = 128
    #checkpoint_dir = 'models'    
    bpp_real, time,  psnr, msssim = compress(input, output, num_filters, checkpoint_dir)
    #decompress(input, output, num_filters, checkpoint_dir)
    #eval_bpp_list.append(eval_bpp)
    bpp_real_list.append(bpp_real)
    time_list.append(time)
    psnr_list.append(psnr)
    msssim_list.append(msssim)
  print("the Encoding processing:")
  print("\n")
  print("RGB PSNR (dB): {:0.2f}".format(np.mean(psnr_list)))
  print("RGB Multiscale SSIM: {:0.4f}".format(np.mean(msssim_list)))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - np.mean(msssim_list))))
  #print("Information content in bpp: {:0.4f}".format(np.mean(eval_bpp_list)))
  print("Actual bits per pixel: {:0.4f}".format(np.mean(bpp_real_list)))
  print("the average time is {:0.4f}".format(np.mean(time_list)))
  
  
  
def decoder_main():
  eval_bpp_list = []
  bpp_real_list = []
  time_list = []
  psnr_list = []
  msssim_list = []

    
  for image_file in glob(save_image_name_path+'/'+'*.npz'):
    print(image_file[:])
    image_name_path = image_file.split('/')[-1]
    input = save_image_name_path+'/'+image_name_path
    output = save_recImage_path+'/'
    origin_image_file = path +  image_name_path[:-4]
    print(input)
    print(output)
    #output = 'images/'
    num_filters = 128
    #checkpoint_dir = 'models'    
    bpp_real, time,  psnr, msssim = decompress(input, output, origin_image_file, num_filters, checkpoint_dir)
    #decompress(input, output, num_filters, checkpoint_dir)
    #eval_bpp_list.append(eval_bpp)
    bpp_real_list.append(bpp_real)
    time_list.append(time)
    psnr_list.append(psnr)
    msssim_list.append(msssim)
  print("\n")
  print("The decoding processing:")
  print("RGB PSNR (dB): {:0.2f}".format(np.mean(psnr_list)))
  print("RGB Multiscale SSIM: {:0.4f}".format(np.mean(msssim_list)))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - np.mean(msssim_list))))
  #print("Information content in bpp: {:0.4f}".format(np.mean(eval_bpp_list)))
  print("Actual bits per pixel: {:0.4f}".format(np.mean(bpp_real_list)))
  print("the average time is {:0.4f}".format(np.mean(time_list)))
  
if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  endcoder_main()
  decoder_main()
