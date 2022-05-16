import pdb
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NumOfCrossValidationFolds = 1
np.random.seed(10)
timestep = 120
SkipPacketsForSampling = 1
IncrementalSampling = False
# NumberOfSamplesUntiIncrement = 10000
# IncrementalStepMultiplier = 1
ComputeInterArrival = True
DescretesizeLength = False
DirectionLengthCombined = True
NormalizeLength = True
NormalizeInterArrival = True
StandardLength = 50
MaxInterArrival = 0.05
Starting_point = 0
StartingPointMultiply = 13
Num_of_extracted_subflow = 100
PaddingEnable = True
PadAtTheBegining = True
PaddingThreshold = 20
CompureStatisticsInThisScript = True

def loadData2(seq_len):
    # seq_len = 25
    seq_len = 25
    df = np.load('../data3.npy',allow_pickle=True)
    label = np.load('../label3.npy',allow_pickle=True)
    col_time = df[:,:seq_len]
    for i in range(col_time.shape[0]):
        for j in range(1,col_time.shape[1]):
            col_time[i][j] = col_time[i][j] - col_time[i][0]
        col_time[i][0] = 0

    col_direct = df[:,30:(30+seq_len)]
    col_direct = np.where(col_direct == 0, -1, 1)

    col_size = df[:,60:(60+seq_len)]
    print(col_time.shape)
    print(col_size.shape)
    print(col_direct.shape)
    col_size_with_direct = col_size * col_direct

    col_size_with_direct = col_size_with_direct / StandardLength
    col_intv_time = col_time
    MaxInterArrival = np.max(col_intv_time)
    ttt = col_intv_time /MaxInterArrival
    ttt = np.where(ttt > 1, 1, ttt)
    ttt = (ttt - 0.5) * 2
    # label
    i = 0
    gen_label = []
    print('--------------------------------')
    for seq_label in label:
        duration = col_time[i][-1] - col_time[i][0]
        bandwidth = np.sum(col_size[i]) / duration
        tmp_label = [duration, bandwidth, seq_label]
        gen_label.append(tmp_label)
        # labellist[i] =
        i += 1
    print('1111111111111111111111111111111')
    gen_label = np.array(gen_label)
    # data = [ttt,col_size_with_direct , col_direct]
    # print(ttt.shape,col_size_with_direct.shape,col_direct.shape)
    data = np.concatenate([ttt,col_size_with_direct,col_direct],axis=1)
    data = np.array(data)
    return (data, gen_label, label)

def loadData1(seq_len):
    seq_len = 25

    df = pd.read_csv('../../fingerprint/data/mixture_10d.csv')
    time = np.array(df['time'])
    label = np.array(df['label'])
    size = np.array(df['size'])
    direct = np.array(df['direct'])
    length = len(time)
    remain_len = length - length % seq_len
    split_nums = remain_len/seq_len

    time = time[:remain_len]
    label = label[:remain_len]
    size = size[:remain_len]
    direct = direct[:remain_len]

    time = np.split(time,split_nums)
    label = np.split(label,split_nums)
    size = np.split(size,split_nums)
    direct = np.split(direct,split_nums)

    time,label,size,direct = np.array(time),np.array(label),np.array(size),np.array(direct)
    for i in range(time.shape[0]):
        for j in range(1,time.shape[1]):
            time[i][j] = time[i][j] - time[i][0]
        time[i][0] = 0

    direct = np.where(direct == 0, -1, 1)
    size_with_direct = size * direct/StandardLength

    MaxInterArrival = np.max(time)
    ttt = time / MaxInterArrival
    ttt = np.where(ttt > 1, 1, ttt)
    ttt = (ttt - 0.5) * 2

    i = 0
    gen_label = []
    truelabel = []
    print('--------------------------------')
    for seq_label in label:
        each_label = [0]*25
        for i in seq_label:
            if i>-1:
                each_label[i] = 1
        duration = time[i][-1] - time[i][0]
        bandwidth = np.sum(size[i]) / duration
        tmp_label = [duration, bandwidth, each_label]
        gen_label.append(tmp_label)
        truelabel.append(each_label)
        # labellist[i] =
        i += 1
    print('1111111111111111111111111111111')
    gen_label = np.array(gen_label)
    # data = [ttt,col_size_with_direct , col_direct]
    # print(ttt.shape,col_size_with_direct.shape,col_direct.shape)
    data = np.concatenate([ttt, size_with_direct, direct], axis=1)
    data = np.array(data)
    return (data, gen_label, truelabel)



def run1():
    (data, label, classlabel) = loadData1(timestep)
    # print(data.shape,label.shape,classlabel.shape)
    train, x_test_val, trainL, y_val_test = train_test_split(data, label, test_size=0.5, random_state=42)
    test, val, testL, valL = train_test_split(x_test_val, y_val_test, test_size=0.3, random_state=42)
    train_, x_test_val_, trainL_, y_val_test_ = train_test_split(data, classlabel, test_size=0.5, random_state=42)
    test_, val_, testL_, valL_ = train_test_split(x_test_val_, y_val_test_, test_size=0.3, random_state=42)

    test = np.concatenate([test,val],axis=0)
    testL = np.concatenate([testL,valL],axis=0)
    testL_ = np.concatenate([testL_,valL_],axis=0)
    np.save("trainData1.npy", train)
    np.save("trainLabel1.npy", trainL)
    np.save("trainLabel_1.npy", trainL_)
    # np.save("valData2.npy", val)
    # np.save("valLabel2.npy", valL)
    # np.save("valLabel_2.npy", valL_)
    np.save("testData1.npy", test)
    np.save("testLabel1.npy", testL)
    np.save("testLabel_1.npy", testL_)
    print(train.shape, trainL.shape)
    # print(val.shape, valL.shape)
    print(test.shape, testL.shape)
def run2():
    (data, label, classlabel) = loadData2(timestep)
    # print(data.shape,label.shape,classlabel.shape)
    train, x_test_val, trainL, y_val_test = train_test_split(data, label, test_size=0.5, random_state=42)
    test, val, testL, valL = train_test_split(x_test_val, y_val_test, test_size=0.3, random_state=42)
    train_, x_test_val_, trainL_, y_val_test_ = train_test_split(data, classlabel, test_size=0.5, random_state=42)
    test_, val_, testL_, valL_ = train_test_split(x_test_val_, y_val_test_, test_size=0.3, random_state=42)

    test = np.concatenate([test,val],axis=0)
    testL = np.concatenate([testL,valL],axis=0)
    testL_ = np.concatenate([testL_,valL_],axis=0)
    np.save("trainData2.npy", train)
    np.save("trainLabel2.npy", trainL)
    np.save("trainLabel_2.npy", trainL_)
    # np.save("valData2.npy", val)
    # np.save("valLabel2.npy", valL)
    # np.save("valLabel_2.npy", valL_)
    np.save("testData2.npy", test)
    np.save("testLabel2.npy", testL)
    np.save("testLabel_2.npy", testL_)
    print(train.shape, trainL.shape)
    # print(val.shape, valL.shape)
    print(test.shape, testL.shape)
run1()
run2()