# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:19:33 2019

@author: Edward
"""
#compute PSNR of two images

import tensorflow as tf
hr_content = tf.read_file('im1.jpg')#recommend : .jpg (.png and .bmp also ok) 
sr_content = tf.read_file('im2.jpg')
hr = tf.image.decode_image(hr_content)
sr = tf.image.decode_image(sr_content)
# read hr image and sr image

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y = sess.run(tf.image.psnr(hr, sr, max_val=255))
    print('PSNR value is ',y)
