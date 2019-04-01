# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:14:15 2019

@author: Edward
"""
#compute PSNR of two matrix (.bin)

import numpy as np
import tensorflow as tf

x1 = np.fromfile('a.bin', dtype=np.float32)#a and b are both 256*256*1
x1 = x1[0:65536]
x1 = x1.reshape(256,256,1)
x2 = np.fromfile('b.bin', dtype=np.float32)
x2 = x2[0:65536]
x2 = x2.reshape(256,256,1)
new_data1=[]
new_data2=[]
for i in range(len(x1)):
  new_data1.append(max(x1[i]))
for i in range(len(x2)):
  new_data2.append(max(x2[i]))
max_numa = max(new_data1)
max_numb = max(new_data2)
max_num = max(max_numa,max_numb)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y = sess.run(tf.image.psnr(x1, x2, max_val=max_num))
    print('PSNR value is', y[0])