# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:57:00 2018

@author: slab
"""

import numpy as np
from matplotlib import pyplot as plt
import f_SOT_model
import SOT_model
#import tradition_LDA as t_LDA

start = 10
topics = start 
end = 1
iteration_num = 5
clip = 10
palpha = 0.05
plam = 0.1
pgamma = 0.00001
pbeta = 0.05
c_len = 15

# TDT2 Dataset
tdt2_data = np.load("tdt2_em_v4_0_100.npy")
tdt2_data = tdt2_data[:clip]
stop_file = open('stopwords2.txt', 'r')
readtext = stop_file.read()
stop_list = readtext.split('\n')
texts = [[word for word in line.lower().split() if word not in stop_list] for line in tdt2_data] 
t_data = texts[:clip]

save_p = "f_SOT_model_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
# RUN
SOT_model.run(t_data, start, end, iteration_num, save_p, clip, plam, c_len, palpha, pbeta, pgamma)
save_p2 = "SOT_model_EXP" + str(clip)+"_" +str(c_len)+"_"+str(palpha)+"_"+str(pbeta)+"_"+str(pgamma)+"\\"
f_SOT_model.run(t_data, start, end, iteration_num, save_p, clip, plam, c_len, palpha, pbeta, pgamma)
