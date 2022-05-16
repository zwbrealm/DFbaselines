import os
import random
import numpy as np
import pandas as pd
import time
n_steps = 25

def loadData2(seq_len):
    labellist = np.load('../label3.npy',allow_pickle=True)
    df = np.load('../data3.npy',allow_pickle=True)
    print('-----------------------')
   

    data = df[:,60:(60+seq_len)]
    np.save(r'data2.npy', data)
    np.save(r'label2.npy', labellist)

def loadData1(seq_len):
    df = pd.read_csv('../../fingerprint/data/mixture_10d.csv')
    length = len(df['size'])
    size = np.array(df['size'])
    label = np.array(df['label'])
     #取整，最后一点不要了
    #对label内容,以seq_len,进行转换
    remain_len = length - length%seq_len
    split_num = remain_len/seq_len
    size = size[:remain_len]
    label = label[:remain_len]
    size = np.array(np.split(size,split_num))
    label = np.array(np.split(label,split_num))

    truelabel = []
    for seq in label:
        each_label = [0] * 25
        for i in seq:
            if i >= 0:
                each_label[i] = 1
        truelabel.append(each_label)



    np.save(r'data1.npy',size,allow_pickle=True)
    np.save(r'label1.npy',truelabel,allow_pickle=True)

if __name__ == "__main__":

    loadData1(n_steps)
