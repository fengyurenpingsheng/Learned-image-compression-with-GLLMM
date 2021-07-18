#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
import tensorflow_compression as tfc
import math
import time
import scipy.special

##new add
import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
from PIL import Image

from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
import tensorflow_probability as  tfp

tfd = tfp.distributions 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def get_image_size(input_file):
  I = Image.open(input_file)
  I_array = np.array(I)
  height_ori, width_ori, _ = np.shape(I_array)
  height = (height_ori // 64) * 64 if height_ori % 64 == 0 else (height_ori // 64 + 1) * 64
  width = (width_ori // 64) * 64 if width_ori % 64 == 0 else (width_ori // 64 + 1) * 64
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  real_height_start = top_end
  real_height_end = top_end + height_ori
  real_width_start = left_end
  real_width_end = left_end + width_ori
  I_array_padded = np.zeros((1,height,width,3), np.uint8)
  I_array_padded[0,top_end:top_end+height_ori, left_end:left_end+width_ori,:]=I_array
  print('height_pad:', height, 'width_pad:', width)
  
  return I_array_padded,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)



def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image

def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

def save_image(filename, image):
  """Saves an image to a PNG file."""

  image = tf.clip_by_value(image, 0, 1)
  image = tf.round(image * 255)
  image = tf.cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)
  
 
def residualblock(tensor, num_filters, scope="residual_block"):
  """Builds the residual block"""
  with tf.variable_scope(scope):
    with tf.variable_scope("conv0"):
      layer = tfc.SignalConv2D(
        num_filters//2, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(tensor)

    with tf.variable_scope("conv1"):
      layer = tfc.SignalConv2D(
        num_filters//2, (3,3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.relu, name='signal_conv2d')
      output = layer(output)

    with tf.variable_scope("conv2"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      output = layer(output)
      
    tensor = tensor + output
       
  return tensor


def NonLocalAttentionBlock(input_x, num_filters, scope="NonLocalAttentionBlock"):
  """Builds the non-local attention block"""
  with tf.variable_scope(scope):
    trunk_branch = residualblock(input_x, num_filters, scope="trunk_RB_0")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_1")
    trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_2")
    
    
    attention_branch = residualblock(input_x, num_filters, scope="attention_RB_0")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_1")
    attention_branch = residualblock(attention_branch, num_filters, scope="attention_RB_2")

    with tf.variable_scope("conv_1x1"):
      layer = tfc.SignalConv2D(
        num_filters, (1,1), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      attention_branch = layer(attention_branch)
    attention_branch = tf.sigmoid(attention_branch)
  
  tensor = input_x + tf.multiply(attention_branch, trunk_branch)
  return tensor

def analysis_transform(tensor, num_filters):
  """Builds the analysis transform."""

  kernel_size = 3
  #Use three 3x3 filters to replace one 9x9
  
  with tf.variable_scope("analysis"):

    # Four down-sampling blocks
    for i in range(4):
      if i > 0:
        with tf.variable_scope("Block_" + str(i) + "_layer_0"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor2 = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_1"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor2 = layer(tensor2)
        
        tensor3 = tensor + tensor2
        
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor4 = layer(tensor3)
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor5 = layer(tensor4)
          
        tensor6 = tensor5 + tensor3
        tensor  = tensor + tensor6

      if i < 3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):
          shortcut = tfc.SignalConv2D(num_filters, (1, 1), corr=True, strides_down=2, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)

        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='gdn'), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

        if i == 1:
          #Add one NLAM
          tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")
          

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=True, strides_down=2, padding="same_zeros",
            use_bias=False, activation=None, name='signal_conv2d') 
          tensor = layer(tensor)

        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")
        

    return tensor

def hyper_analysis(tensor, num_filters):
  """Build the analysis transform in hyper"""

  with tf.variable_scope("hyper_analysis"):
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters     
    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters 
    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=True, strides_down=2, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters):
  """Builds the synthesis transform."""

  kernel_size = 3
  #Use four 3x3 filters to replace one 9x9
  
  with tf.variable_scope("synthesis"):

    # Four up-sampling blocks
    for i in range(4):
      if i == 0:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")

      if i == 2:
        #Add one NLAM
        tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")
        
      with tf.variable_scope("Block_" + str(i) + "_layer_0"):
        layer = tfc.SignalConv2D(
          num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        tensor2 = layer(tensor)

      with tf.variable_scope("Block_" + str(i) + "_layer_1"):
        layer = tfc.SignalConv2D(
          num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
        tensor2 = layer(tensor2)
        #tensor = tensor + tensor2
        
        
        
        tensor3 = tensor + tensor2
        
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor4 = layer(tensor3)
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
              num_filters, (kernel_size, kernel_size), corr=True, strides_down=1, padding="same_zeros",
              use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor5 = layer(tensor4)
          
        tensor6 = tensor5 + tensor3
        tensor  = tensor + tensor6
        
        


      if i <3:
        with tf.variable_scope("Block_" + str(i) + "_shortcut"):

          # Use Sub-Pixel to replace deconv.
          shortcut = tfc.SignalConv2D(num_filters*4, (1, 1), corr=False, strides_up=1, padding="same_zeros",
                                      use_bias=True, activation=None, name='signal_conv2d')
          shortcut_tensor = shortcut(tensor)
          shortcut_tensor = tf.depth_to_space(shortcut_tensor, 2)

        with tf.variable_scope("Block_" + str(i) + "_layer_2"):

          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(num_filters*4, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)         
          
        with tf.variable_scope("Block_" + str(i) + "_layer_3"):
          layer = tfc.SignalConv2D(
            num_filters, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(name='igdn', inverse=True), name='signal_conv2d')
          tensor = layer(tensor)
          
          tensor = tensor + shortcut_tensor

      else:
        with tf.variable_scope("Block_" + str(i) + "_layer_2"):
          
          # Use Sub-Pixel to replace deconv.
          layer = tfc.SignalConv2D(12, (kernel_size, kernel_size), corr=False, strides_up=1, padding="same_zeros",
                                   use_bias=True, activation=None, name='signal_conv2d')
          tensor = layer(tensor)
          tensor = tf.depth_to_space(tensor, 2)
          

    return tensor

def hyper_synthesis(tensor, num_filters):
  """Builds the hyper synthesis transform"""

  with tf.variable_scope("hyper_synthesis", reuse=tf.AUTO_REUSE):
    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
        num_filters, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    #One 5x5 is replaced by two 3x3 filters
    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_3"):
      layer = tfc.SignalConv2D(
        num_filters*1.5, (3, 3), corr=False, strides_up = 2, padding="same_zeros",
        use_bias=True, activation=tf.nn.leaky_relu, name='signal_conv2d')
      tensor = layer(tensor)

    with tf.variable_scope("layer_4"):
      layer = tfc.SignalConv2D(
        num_filters*2, (3, 3), corr=False, strides_up = 1, padding="same_zeros",
        use_bias=True, activation=None, name='signal_conv2d')
      tensor = layer(tensor)

    return tensor

def masked_conv2d(
    inputs,
    num_outputs,
    kernel_shape, # [kernel_height, kernel_width]
    mask_type, # None, "A" or "B",
    strides=[1, 1], # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    scope="masked"):
  
  with tf.variable_scope(scope):
    mask_type = mask_type.lower()
    batch_size, height, width, channel = inputs.get_shape().as_list()

    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
      "kernel height and width should be odd number"

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    weights_shape = [kernel_h, kernel_w, channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, weights_initializer, weights_regularizer)

    if mask_type is not None:
      mask = np.ones(
        (kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

      mask[center_h, center_w+1: ,: ,:] = 0.
      mask[center_h+1:, :, :, :] = 0.

      if mask_type == 'a':
        mask[center_h,center_w,:,:] = 0.

      weights *= tf.constant(mask, dtype=tf.float32)
      tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

    outputs = tf.nn.conv2d(inputs,
        weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
    tf.add_to_collection('conv2d_outputs', outputs)

    if biases_initializer != None:
      biases = tf.get_variable("biases", [num_outputs,],
          tf.float32, biases_initializer, biases_regularizer)
      outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

    if activation_fn:
      outputs = activation_fn(outputs, name='outputs_with_fn')

    return outputs

def entropy_parameter(tensor, inputs, num_filters, training):
  """tensor: the output of hyper autoencoder (phi) to generate the mean and variance
     inputs: the variable needs to be encoded. (y)
  """
  with tf.variable_scope("entropy_parameter", reuse=tf.AUTO_REUSE):

    half = tf.constant(.5)

    if training:
      noise = tf.random_uniform(tf.shape(inputs), -half, half)
      values = tf.add_n([inputs, noise])

    else: #inference
      #if inputs is not None: #compress
      values = tf.round(inputs)
        

    masked = masked_conv2d(values, num_filters*2, [5, 5], "A", scope='masked')
    tensor = tf.concat([masked, tensor], axis=3)
      

    with tf.variable_scope("layer_0"):
      layer = tfc.SignalConv2D(
          640, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_1"):
      layer = tfc.SignalConv2D(
          640*2, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=True, activation=tf.nn.leaky_relu)
      tensor = layer(tensor)

    with tf.variable_scope("layer_2"):
      layer = tfc.SignalConv2D(
          num_filters*30, (1, 1), corr=True, strides_down=1, padding="same_zeros",
          use_bias=False, activation=None)
      tensor = layer(tensor)


    #=========Gaussian Mixture Model=========
    prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2, prob3, mean3, scale3, prob4, mean4, scale4, prob5, mean5, scale5, \
    prob6, mean6, scale6, prob7, mean7, scale7, prob8, mean8, scale8, prob_m0, prob_m1, prob_m2= \
      tf.split(tensor, num_or_size_splits=30, axis=3)
    scale0 = tf.abs(scale0)
    scale1 = tf.abs(scale1)
    scale2 = tf.abs(scale2)
    scale3 = tf.abs(scale3)
    scale4 = tf.abs(scale4)
    scale5 = tf.abs(scale5)
    scale6 = tf.abs(scale6)
    scale7 = tf.abs(scale7)
    scale8 = tf.abs(scale8)

    probs = tf.stack([prob0, prob1, prob2], axis=-1)
    probs = tf.nn.softmax(probs, axis=-1)
    probs_lap = tf.stack([prob3, prob4, prob5], axis=-1)
    probs_lap = tf.nn.softmax(probs_lap, axis=-1)
    probs_log = tf.stack([prob6, prob7, prob8], axis=-1)
    probs_log = tf.nn.softmax(probs_log, axis=-1)
    probs_mix = tf.stack([prob_m0, prob_m1, prob_m2], axis=-1)
    probs_mix = tf.nn.softmax(probs_mix, axis=-1)

    # To merge them together
    means = tf.stack([mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8], axis=-1)
    variances = tf.stack([scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7, scale8], axis=-1)

    # =======================================

    #Calculate the likelihoods for inputs
    #if inputs is not None:
    if training:
      dist_0 = tfd.Normal(loc = mean0, scale = scale0, name='dist_0')
      dist_1 = tfd.Normal(loc = mean1, scale = scale1, name='dist_1')
      dist_2 = tfd.Normal(loc = mean2, scale = scale2, name='dist_2')
      dist_3 = tfd.Laplace(loc=mean3, scale=scale3, name='dist_3')
      dist_4 = tfd.Laplace(loc=mean4, scale=scale4, name='dist_4')
      dist_5 = tfd.Laplace(loc=mean5, scale=scale5, name='dist_5')
      dist_6 = tfd.Logistic(loc=mean6, scale=scale6, name='dist_6')
      dist_7 = tfd.Logistic(loc=mean7, scale=scale7, name='dist_7')
      dist_8 = tfd.Logistic(loc=mean8, scale=scale8, name='dist_8')

      #=========Gaussian Mixture Model=========
      likelihoods_0 = dist_0.cdf(values + half) - dist_0.cdf(values - half)
      likelihoods_1 = dist_1.cdf(values + half) - dist_1.cdf(values - half)
      likelihoods_2 = dist_2.cdf(values + half) - dist_2.cdf(values - half)
      likelihoods_3 = dist_3.cdf(values + half) - dist_3.cdf(values - half)
      likelihoods_4 = dist_4.cdf(values + half) - dist_4.cdf(values - half)
      likelihoods_5 = dist_5.cdf(values + half) - dist_5.cdf(values - half)
      likelihoods_6 = dist_6.cdf(values + half) - dist_6.cdf(values - half)
      likelihoods_7 = dist_7.cdf(values + half) - dist_7.cdf(values - half)
      likelihoods_8 = dist_8.cdf(values + half) - dist_8.cdf(values - half)

      likelihoods = probs_mix[:,:,:,:,0]*(probs[:,:,:,:,0]*likelihoods_0 + probs[:,:,:,:,1]*likelihoods_1 + probs[:,:,:,:,2]*likelihoods_2) + \
                    probs_mix[:,:,:,:,1]*(probs_lap[:, :, :, :, 0] * likelihoods_3 + probs_lap[:, :, :, :, 1] * likelihoods_4 + probs_lap[:, :, :, :,2] * likelihoods_5) + \
                    probs_mix[:, :, :, :, 2] * (probs_log[:, :, :, :, 0] * likelihoods_6 + probs_log[:, :, :, :,1] * likelihoods_7 + probs_log[:, :, :,:,2] * likelihoods_8)


      # =======REVISION: Robust version ==========
      edge_min = probs_mix[:,:,:,:,0]*(probs[:,:,:,:,0]*dist_0.cdf(values + half) +
                 probs[:,:,:,:,1]*dist_1.cdf(values + half) +
                 probs[:,:,:,:,2]*dist_2.cdf(values + half)) + \
                 probs_mix[:,:,:,:,1]*(probs_lap[:, :, :, :, 0] * dist_3.cdf(values + half) +
                 probs_lap[:, :, :, :, 1] * dist_4.cdf(values + half) +
                 probs_lap[:, :, :, :, 2] * dist_5.cdf(values + half)) + \
                 probs_mix[:, :, :, :, 2] * (probs_log[:, :, :, :, 0] * dist_6.cdf(values + half) +
                                             probs_log[:, :, :, :, 1] * dist_7.cdf(values + half) +
                                             probs_log[:, :, :, :, 2] * dist_8.cdf(values + half))

      edge_max = probs_mix[:,:,:,:,0]*(probs[:,:,:,:,0]* (1.0 - dist_0.cdf(values - half)) +
                 probs[:,:,:,:,1]* (1.0 - dist_1.cdf(values - half)) +
                 probs[:,:,:,:,2]* (1.0 - dist_2.cdf(values - half))) + \
                 probs_mix[:,:,:,:,1]*(probs_lap[:, :, :, :, 0] * (1.0 - dist_3.cdf(values - half)) +
                 probs_lap[:, :, :, :, 1] * (1.0 - dist_4.cdf(values - half)) +
                 probs_lap[:, :, :, :, 2] * (1.0 - dist_5.cdf(values - half))) + \
                 probs_mix[:, :, :, :, 2] * (probs_log[:, :, :, :, 0] * (1.0 - dist_6.cdf(values - half)) +
                                             probs_log[:, :, :, :, 1] * (1.0 - dist_7.cdf(values - half)) +
                                             probs_log[:, :, :, :, 2] * (1.0 - dist_8.cdf(values - half)))
      likelihoods = tf.where(values < -254.5, edge_min, tf.where(values > 255.5, edge_max, likelihoods))


      likelihood_lower_bound = tf.constant(1e-6)
      likelihood_upper_bound = tf.constant(1.0)
      likelihoods = tf.minimum(tf.maximum(likelihoods, likelihood_lower_bound), likelihood_upper_bound)

    else:
      #values = None
      likelihoods = None

  return values, likelihoods, means, variances, probs, probs_lap, probs_log, probs_mix

def compress(input, output, num_filters, checkpoint_dir):

    start = time.time()
    tf.set_random_seed(1)
    tf.reset_default_graph()
      
      
      #with tf.device('/cpu:0'):
        # Load input image and add batch dimension.
        
    #x = load_image(input)
    #print("x shape is {}".format(x.get_shape().as_list()))
    images_info = get_image_size(input)
    images_padded_numpy, size = images_info
    print("the size is {}".format(len(size)))
    real_height_start, real_height_end, real_width_start, real_width_end, height, width = size
    with tf.name_scope('Data'):
      images_padded = tf.placeholder(tf.float32, shape=(1, height, width, 3), name='images_ori')
      x = images_padded
      x_shape = tf.shape(x)

    y = analysis_transform(x, num_filters)

    # Build a hyper autoencoder
    z = hyper_analysis(y, num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()
    string = entropy_bottleneck.compress(z)
    string = tf.squeeze(string, axis=0)

    z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

    # To decompress the z_tilde back to avoid the inconsistence error
    string_rec = tf.expand_dims(string, 0)
    z_tilde = entropy_bottleneck.decompress(string_rec, tf.shape(z)[1:], channels=num_filters)

    phi = hyper_synthesis(z_tilde, num_filters)


    # REVISION： for Gaussian Mixture Model (GMM), use window-based fast implementation    
    #y = tf.clip_by_value(y, -255, 256)
    y_hat = tf.round(y)


    tiny_y = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters])
    tiny_phi = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters*2]) 
    #_, _, y_means, y_variances, y_probs = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)
    _, _, y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)

    x_hat = synthesis_transform(y_hat, num_filters)


    num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
    #x_hat = x_hat[0, :tf.shape(x)[1], :tf.shape(x)[2], :]

    #op = save_image('temp/temp.png', x_hat)

    # Mean squared error across pixels.
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    x_ori = x[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
    x_hat = x_hat[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
    mse = tf.reduce_mean(tf.squared_difference(x_ori, x_hat))


    with tf.Session() as sess:
      #print(tf.trainable_variables())
      sess.run(tf.global_variables_initializer())
      # Load the latest model checkpoint, get the compressed string and the tensor
      # shapes.
      #latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
      
      # latest = "models/model-1399000" #lambda = 14
        
      # print(latest)
      # tf.train.Saver().restore(sess, save_path=latest)
      
      
      vars_restore = [var for var in tf.global_variables()]
      saver_0 = tf.train.Saver(vars_restore)
      print(f'Loading learned model from checkpoint {checkpoint_dir}')
      saver_0.restore(sess, checkpoint_dir)

      
      
      string, x_shape, y_shape, num_pixels, y_hat_value, phi_value = \
              sess.run([string, tf.shape(x), tf.shape(y), num_pixels, y_hat, phi],feed_dict={images_padded:images_padded_numpy/255.0})
      

      
      minmax = np.maximum(abs(y_hat_value.max()), abs(y_hat_value.min()))
      minmax = int(np.maximum(minmax, 1))
      #num_symbols = int(2 * minmax + 3)
      print(minmax)
      #print(num_symbols)
      
      # Fast implementations by only encoding non-zero channels with 128/8 = 16bytes overhead
      flag = np.zeros(y_shape[3], dtype=np.int)
      
      for ch_idx in range(y_shape[3]):
        if np.sum(abs(y_hat_value[:, :,:, ch_idx])) > 0:
          flag[ch_idx] = 1

      non_zero_idx = np.squeeze(np.where(flag == 1))

                 
      num = np.packbits(np.reshape(flag, [8, y_shape[3]//8]))
           
      # ============== encode the bits for z===========
      if os.path.exists(output):
        os.remove(output)

      fileobj = open(output, mode='wb')
      fileobj.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
      fileobj.write(np.array([len(string), minmax], dtype=np.uint16).tobytes())
      fileobj.write(np.array(num, dtype=np.uint8).tobytes())
      fileobj.write(string)
      fileobj.close()



      # ============ encode the bits for y ==========
      print("INFO: start encoding y")
      encoder = RangeEncoder(output[:-4] + '.bin')
      samples = np.arange(0, minmax*2+1)
      TINY = 1e-10

       

      kernel_size = 5
      pad_size = (kernel_size - 1)//2
      
      
      
      padded_y = np.pad(y_hat_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))
      padded_phi = np.pad(phi_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))

      
      for h_idx in range(y_shape[1]):
        for w_idx in range(y_shape[2]):          

          
          extracted_y = padded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]
          extracted_phi = padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]

          
          # y_means_values, y_variances_values, y_probs_values = \
                          # sess.run([y_means, y_variances, y_probs], \
                                   # feed_dict={images_padded:images_padded_numpy/255.0, tiny_y: extracted_y, tiny_phi: extracted_phi})         
                                   
          y_means_values, y_variances_values, y_probs_values, y_probs_lap_values, y_probs_log_values, y_probs_mix_values = \
                          sess.run([y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix], \
                                   feed_dict={tiny_y: extracted_y, tiny_phi: extracted_phi})  

          
          
          for i in range(len(non_zero_idx)):
            ch_idx = non_zero_idx[i]
            
            # mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            # sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            # weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]
            
            mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]
            weight_lap = y_probs_lap_values[0, pad_size, pad_size, ch_idx, :]
            weight_log = y_probs_log_values[0, pad_size, pad_size, ch_idx, :]
            weight_mix = y_probs_mix_values[0, pad_size, pad_size, ch_idx, :]

            start00 = time.time()

            # Calculate the pmf/cdf            
            # pmf = (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                  # (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] +\
                  # (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]
            # pmf = weight[0]*((scipy.stats.logistic.cdf(samples + 0.5, loc=mu[0], scale=sigma[0] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[0], scale=sigma[0] + TINY))) + \
                  # (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[1], scale=sigma[1] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[1], scale=sigma[1] + TINY)) * weight[1] + \
                  # (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[2], scale=sigma[2] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[2], scale=sigma[2] + TINY)) * weight[2]
                  
                  
            pmf = weight_mix[0]*((scipy.stats.norm.cdf(samples + 0.5, loc=mu[0], scale=sigma[0] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[0], scale=sigma[0] + TINY)) * weight[0] + \
                  (scipy.stats.norm.cdf(samples + 0.5, loc=mu[1], scale=sigma[1] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[1], scale=sigma[1] + TINY)) * weight[1] + \
                  (scipy.stats.norm.cdf(samples + 0.5, loc=mu[2], scale=sigma[2] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[2], scale=sigma[2] + TINY)) * weight[2]) + \
                  weight_mix[1]*((scipy.stats.laplace.cdf(samples + 0.5, loc=mu[3], scale=sigma[3] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[3], scale=sigma[3] + TINY)) * weight_lap[0] + \
                  (scipy.stats.laplace.cdf(samples + 0.5, loc=mu[4], scale=sigma[4] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[4], scale=sigma[4] + TINY)) * weight_lap[1] + \
                  (scipy.stats.laplace.cdf(samples + 0.5, loc=mu[5], scale=sigma[5] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[5], scale=sigma[5] + TINY)) * weight_lap[2]) + \
                  weight_mix[2]*((scipy.stats.logistic.cdf(samples + 0.5, loc=mu[6], scale=sigma[6] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[6], scale=sigma[6] + TINY)) * weight_log[0] + \
                  (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[7], scale=sigma[7] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[7], scale=sigma[7] + TINY)) * weight_log[1] + \
                  (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[8], scale=sigma[8] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[8], scale=sigma[8] + TINY)) * weight_log[2])
            '''
            # Add the tail mass
            pmf[0] += 0.5 * (1 + scipy.special.erf(( -0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) * weight[0] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) * weight[1] + \
                      0.5 * (1 + scipy.special.erf(( -0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) * weight[2]
                      
            pmf[-1] += (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] + \
                       (1. - 0.5 * (1 + scipy.special.erf((minmax*2 + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]
            '''
            
            # To avoid the zero-probability            
            pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
            pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
            cdf = list(np.add.accumulate(pmf_clip))
            cdf = [0] + [int(i) for i in cdf]
                      
            symbol = np.int(y_hat_value[0, h_idx, w_idx, ch_idx] + minmax )
            encoder.encode([symbol], cdf)


            

      encoder.close()

      size_real = os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin')
      
      bpp_real = (os.path.getsize(output) + os.path.getsize(output[:-4] + '.bin'))* 8 / num_pixels
      bpp_side = (os.path.getsize(output))* 8 / num_pixels
      

      end = time.time()
      print("Time : {:0.3f}".format(end-start))

      psnr = sess.run(tf.image.psnr(x_hat, x_ori*255, 255),feed_dict={images_padded:images_padded_numpy/255.0})
      msssim = sess.run(tf.image.ssim_multiscale(x_hat, x_ori*255, 255),feed_dict={images_padded:images_padded_numpy/255.0})
      
      print("Actual bits per pixel for this image: {:0.4}".format(bpp_real))
      print("Side bits per pixel for z: {:0.4}".format(bpp_side))
      print("PSNR (dB) : {:0.4}".format(psnr[0]))
      print("MS-SSIM : {:0.4}".format(msssim[0]))
      
      
      return bpp_real, (end-start), psnr[0], msssim[0]

def decompress(input, output, origin_image_input, num_filters, checkpoint_dir):
    start = time.time()
    tf.set_random_seed(1)
    tf.reset_default_graph()

      #with tf.device('/cpu:0'):

    print(input)

    # Read the shape information and compressed string from the binary file.
    fileobj = open(input, mode='rb')
    x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)
    length, minmax = np.frombuffer(fileobj.read(4), dtype=np.uint16)
    num = np.frombuffer(fileobj.read(16), dtype=np.uint8)
    string = fileobj.read(length)

    fileobj.close()

    flag = np.unpackbits(num)
    non_zero_idx = np.squeeze(np.where(flag == 1))


    # Get x_pad_shape, y_shape, z_shape
    pad_size = 64
    x_pad_shape = [1] + [int(math.ceil(x_shape[0] / pad_size) * pad_size)] + [int(math.ceil(x_shape[1] / pad_size) * pad_size)]  + [3]
    y_shape = [1] + [x_pad_shape[1]//16] + [x_pad_shape[2]//16] + [num_filters]
    z_shape = [y_shape[1]//4] + [y_shape[2]//4] + [num_filters]

    ###add the origin image
    origin_image_input = Image.open(origin_image_input)
    origin_Image_array = np.array(origin_image_input)
    origin_Image_array_input = tf.placeholder(tf.float32, shape=list(np.shape(origin_Image_array)), name='origin_Image_array_input')
    
    # Add a batch dimension, then decompress and transform the image back.
    strings = tf.expand_dims(string, 0)

    entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
    z_tilde = entropy_bottleneck.decompress(
        strings, z_shape, channels=num_filters)
    phi = hyper_synthesis(z_tilde, num_filters)

    # Transform the quantized image back (if requested).
    tiny_y = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters])
    tiny_phi = tf.placeholder(dtype=tf.float32, shape= [1] + [5] + [5] + [num_filters*2])
    #_, _, means, variances, probs = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)
    _, _, y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix = entropy_parameter(tiny_phi, tiny_y, num_filters, training=False)

    # Decode the x_hat usign the decoded y
    y_hat = tf.placeholder(dtype=tf.float32, shape=y_shape)
    x_hat = synthesis_transform(y_hat, num_filters)

    x_padding_shape = tf.shape(x_hat)
    # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
    x_hat = x_hat[0, :int(x_shape[0]), :int(x_shape[1]), :]
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    # Write reconstructed image out as a PNG file.
    #op = save_image(output, x_hat)
       
    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
      #latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
      
      
      # latest = "models/model-1399000" #lambda = 14
      # print(latest)

        
      # tf.train.Saver().restore(sess, save_path=latest)
      
      vars_restore = [var for var in tf.global_variables()]
      saver_0 = tf.train.Saver(vars_restore)
      print(f'Loading learned model from checkpoint {checkpoint_dir}')
      saver_0.restore(sess, checkpoint_dir)
      
      
      
      phi_value = sess.run(phi)

      print("INFO: start decoding y")
      print(time.time() - start)


      decoder = RangeDecoder(input[:-4] + '.bin')
      samples = np.arange(0, minmax*2+1)
      TINY = 1e-10

      
      # Fast implementation to decode the y_hat
      kernel_size = 5
      pad_size = (kernel_size - 1)//2

      decoded_y = np.zeros([1] + [y_shape[1]+kernel_size-1] + [y_shape[2]+kernel_size-1] + [num_filters])
      padded_phi = np.pad(phi_value, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant',
                                 constant_values=((0., 0.), (0., 0.), (0., 0.), (0., 0.)))
      


      for h_idx in range(y_shape[1]):
        for w_idx in range(y_shape[2]):



          # y_means, y_variances, y_probs = \
                   # sess.run([means, variances, probs], \
                            # feed_dict={tiny_y: decoded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :], \
                                       # tiny_phi: padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]})
                                       
                                       
                                       
          y_means_values, y_variances_values, y_probs_values, y_probs_lap_values, y_probs_log_values, y_probs_mix_values = \
                          sess.run([y_means, y_variances, y_probs, y_probs_lap, y_probs_log, y_probs_mix], \
                                   feed_dict={tiny_y: decoded_y[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :], \
                                       tiny_phi: padded_phi[:, h_idx: h_idx+kernel_size, w_idx:w_idx+kernel_size, :]})  
         

              
          for i in range(len(non_zero_idx)):
            ch_idx = non_zero_idx[i]

              
            # mu = y_means[0, pad_size, pad_size, ch_idx, :] + minmax
            # sigma = y_variances[0, pad_size, pad_size, ch_idx, :]
            # weight = y_probs[0, pad_size, pad_size, ch_idx, :]
            
            mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
            sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
            weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]
            weight_lap = y_probs_lap_values[0, pad_size, pad_size, ch_idx, :]
            weight_log = y_probs_log_values[0, pad_size, pad_size, ch_idx, :]
            weight_mix = y_probs_mix_values[0, pad_size, pad_size, ch_idx, :]


            # pmf = (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[0]) / ((sigma[0] + TINY) * 2 ** 0.5)))) * weight[0] + \
                  # (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[1]) / ((sigma[1] + TINY) * 2 ** 0.5)))) * weight[1] +\
                  # (0.5 * (1 + scipy.special.erf((samples + 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5))) - \
                   # 0.5 * (1 + scipy.special.erf((samples - 0.5 - mu[2]) / ((sigma[2] + TINY) * 2 ** 0.5)))) * weight[2]
            pmf = weight_mix[0]*((scipy.stats.norm.cdf(samples + 0.5, loc=mu[0], scale=sigma[0] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[0], scale=sigma[0] + TINY)) * weight[0] + \
                  (scipy.stats.norm.cdf(samples + 0.5, loc=mu[1], scale=sigma[1] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[1], scale=sigma[1] + TINY)) * weight[1] + \
                  (scipy.stats.norm.cdf(samples + 0.5, loc=mu[2], scale=sigma[2] + TINY)-scipy.stats.norm.cdf(samples - 0.5, loc=mu[2], scale=sigma[2] + TINY)) * weight[2]) + \
                  weight_mix[1]*((scipy.stats.laplace.cdf(samples + 0.5, loc=mu[3], scale=sigma[3] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[3], scale=sigma[3] + TINY)) * weight_lap[0] + \
                  (scipy.stats.laplace.cdf(samples + 0.5, loc=mu[4], scale=sigma[4] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[4], scale=sigma[4] + TINY)) * weight_lap[1] + \
                  (scipy.stats.laplace.cdf(samples + 0.5, loc=mu[5], scale=sigma[5] + TINY)-scipy.stats.laplace.cdf(samples - 0.5, loc=mu[5], scale=sigma[5] + TINY)) * weight_lap[2]) + \
                  weight_mix[2]*((scipy.stats.logistic.cdf(samples + 0.5, loc=mu[6], scale=sigma[6] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[6], scale=sigma[6] + TINY)) * weight_log[0] + \
                  (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[7], scale=sigma[7] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[7], scale=sigma[7] + TINY)) * weight_log[1] + \
                  (scipy.stats.logistic.cdf(samples + 0.5, loc=mu[8], scale=sigma[8] + TINY)-scipy.stats.logistic.cdf(samples - 0.5, loc=mu[8], scale=sigma[8] + TINY)) * weight_log[2])

            pmf_clip = np.clip(pmf, 1.0/65536, 1.0)
            pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
            cdf = list(np.add.accumulate(pmf_clip))
            cdf = [0] + [int(i) for i in cdf]

            decoded_y[0, h_idx+pad_size, w_idx+pad_size, ch_idx] = decoder.decode(1, cdf)[0] - minmax 

      decoded_y = decoded_y[:, pad_size:y_shape[1]+pad_size, pad_size:y_shape[2]+pad_size, :]
                                                          
      #sess.run(op, feed_dict={y_hat: decoded_y})
      
      num_pixels = tf.to_float(tf.reduce_prod(x_padding_shape[:-1]))
      num_pixels_ = sess.run(num_pixels)
      size_real = os.path.getsize(input) + os.path.getsize(input[:-4] + '.bin')
      bpp_real = (os.path.getsize(input) + os.path.getsize(input[:-4] + '.bin'))* 8 / num_pixels_
      bpp_side = (os.path.getsize(input))* 8 / num_pixels_
      
      end = time.time()
      #print("Time (s): {:0.3f}".format(end-start))
      print("the num_pixels is {:0.4f}".format(num_pixels_))
      print("Time : {:0.3f}".format(end-start))
      #origin_Image_array_input = tf.placeholder(tf.float32, shape=list(np.shape(origin_Image_array)), name='origin_Image_array_input')

      psnr = sess.run(tf.image.psnr(x_hat, origin_Image_array_input*255, 255),feed_dict={origin_Image_array_input:origin_Image_array/255.0, y_hat: decoded_y})
      msssim = sess.run(tf.image.ssim_multiscale(x_hat, origin_Image_array_input*255, 255),feed_dict={origin_Image_array_input:origin_Image_array/255.0, y_hat: decoded_y})
      
      print("Actual bits per pixel for this image: {:0.4}".format(bpp_real))
      print("Side bits per pixel for z: {:0.4}".format(bpp_side))
      print("PSNR (dB) : {:0.4}".format(psnr))
      print("MS-SSIM : {:0.4}".format(msssim))
      
      
      return bpp_real, (end-start), psnr, msssim


def overall_performance(metrics_list):
  psnr_rgb_list = []
  psnr_y_list = []
  psnr_u_list = []
  psnr_v_list = []
  msssim_rgb_list = []
  msssim_y_list = []
  bpp_list = []
  for metrics_item in metrics_list:
    psnr_rgb_list.append(metrics_item[0])
    psnr_y_list.append(metrics_item[1][0])
    psnr_u_list.append(metrics_item[1][1])
    psnr_v_list.append(metrics_item[1][2])
    msssim_rgb_list.append(metrics_item[2])
    msssim_y_list.append(metrics_item[3])
    bpp_list.append(metrics_item[4])
  bpp_avg = np.mean(bpp_list)
  RGB_MSE_avg = np.mean([255. ** 2 / pow(10, PSNR / 10) for PSNR in psnr_rgb_list])
  RGB_PSNR_avg = 10 * np.log10(255. ** 2 / RGB_MSE_avg)
  Y_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_y_list])
  Y_PSNR_avg = 10 * np.log10(255 ** 2 / Y_MSE_avg)
  U_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_u_list])
  U_PSNR_avg = 10 * np.log10(255 ** 2 / U_MSE_avg)
  V_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_v_list])
  V_PSNR_avg = 10 * np.log10(255 ** 2 / V_MSE_avg)
  yuv_psnr_avg = 6.0/8.0*Y_PSNR_avg + 1.0/8.0*U_PSNR_avg + 1.0/8.0*V_PSNR_avg
  msssim_rgb_avg = np.mean(msssim_rgb_list)
  msssim_y_avg = np.mean(msssim_y_list)

  print("overall performance")
  print("RGB PSNR (dB): {:0.2f}".format(RGB_PSNR_avg))
  print("YUV444 PSNR (dB): {:0.2f}".format(yuv_psnr_avg))
  print("RGB Multiscale SSIM: {:0.4f}".format(msssim_rgb_avg))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_rgb_avg)))
  print("Y Multiscale SSIM: {:0.4f}".format(msssim_y_avg))
  print("Y Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_y_avg)))
  print("Actual bits per pixel: {:0.4f}\n".format(bpp_avg))



def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument("--autoregressive", "-AR", action="store_true", help="Include autoregressive model for training")
  parser.add_argument("--num_filters", type=int, default=192, help="Number of filters per layer.")
  parser.add_argument("--restore_path", default=None, help="Directory where to load model checkpoints.")
  parser.add_argument("--checkpoint_dir", default="train", help="Directory where to save/load model checkpoints.")
  parser.add_argument("--if_weight", type=int, default=0.0, help="weights")
  subparsers = parser.add_subparsers(title="commands", dest="command",
      help="commands: 'train' loads training data and trains (or continues "
           "to train) a new model. 'encode' reads an image file (lossless "
           "PNG format) and writes a encoded binary file. 'decode' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Trains (or continues to train) a new model.")
  train_cmd.add_argument("--train_root_dir", default="images", help="The root directory of training data, which contains a list of RGB images in PNG format.")
  train_cmd.add_argument("--batchsize", type=int, default=8, help="Batch size for training.")
  train_cmd.add_argument("--patchsize", type=int, default=256, help="Size of image patches for training.")
  train_cmd.add_argument("--lossWeight", type=float, default=0, dest="lossWeight", help="Weight for MSE-SSIM tradeoff.")
  train_cmd.add_argument("--lambda", type=float, default=0.01, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument("--last_step", type=int, default=1500000, help="Train up to this number of steps.")
  train_cmd.add_argument("--lr", type=float, default = 1e-4, help="Learning rate [1e-4].")
  train_cmd.add_argument("--lr_scheduling", "-lr_sch", action="store_true", help="Enable learning rate scheduling, [enabled] as default")
  train_cmd.add_argument("--preprocess_threads", type=int, default=16, help="Number of CPU threads to use for parallel decoding of training images.")

  # 'encode' subcommand.
  encode_cmd = subparsers.add_parser("encode", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Reads a PNG file, encode it, and writes a 'bitstream' file.")
  # 'decode' subcommand.
  decode_cmd = subparsers.add_parser("decode",formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Reads a 'bitstream' file, reconstructs the image, and writes back a PNG file.")

  # Arguments for both 'encode' and 'decode'.
  for cmd, ext in ((encode_cmd, ".bitstream"), (decode_cmd, ".png")):
    cmd.add_argument("input_file", help="Input filename.")
    cmd.add_argument("output_file", nargs="?", help="Output filename (optional). If not provided, appends '{}' to the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args

def main(args):
  # Invoke subcommand.
  #os.environ['CUDA_VISIBLE_DEVICES'] = "2"
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
  if args.command == "train":
    train(args)
  elif args.command == "encode": # encoding
    if not args.output_file:
      args.output_file = args.input_file + ".bitstream"
    if os.path.isdir(args.input_file):
      dirs = os.listdir(args.input_file)
      test_files = []
      for dir in dirs:
        path = os.path.join(args.input_file, dir)
        if os.path.isdir(path):
          test_files += glob.glob(path + '/*.png')[:6]
        if os.path.isfile(path):
          test_files.append(path)
      if not test_files:
        raise RuntimeError(
          "No testing images found with glob '{}'.".format(args.input_file))
      print("Number of images for testing:", len(test_files))
      metrics_list=[]
      for file_idx in range(len(test_files)):
        file = test_files[file_idx]
        print(str(file_idx)+" testing image:", file)
        args.input_file = file
        #file_name = file.split('/')[-1]
        #args.output_file = args.output_file + file_name.replace('.png', '.bitstream')
        image_padded, size = get_image_size(args.input_file)
        metrics = encode(args, image_padded, size)
        metrics_list.append(metrics)
      overall_performance(metrics_list)
    else:
      image_padded, size = get_image_size(args.input_file)
      metrics = encode(args, image_padded, size, True)
  elif args.command == "decode": # decoding
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decode(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)


