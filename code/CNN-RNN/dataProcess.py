import numpy as np
import pandas as pd
import time

# def read_csv(file_path, has_header=True):
#     data = pd.read_csv(file_path)
#     data = pd.DataFrame(data,columns=['time','size','direct','label'])
#     return data
#
# def data_fe(data, seq_len, label_len):
#     feature = []
#     label = []
#     prev = 0
#     lv = [0] * label_len
#
#     for i, row in data.iterrows():
#         # a sequence feature vector
#         fv = []
#         fv.append(row['size'])
#         # inter-time: first = 0, rest = curr - prev
#         if prev == 0.0:
#             fv.append(0.0)
#         else:
#             fv.append(float(row['time']) - float(prev))
#             if i<30:
#                 print(float(row['time'] - float(prev)))
#
#         prev = float(row['time'])
#         fv.append(row['direct'])
#         feature.append(fv)
#         try:
#             lv[int(row['label'])] = 1
#         except ValueError:
#             print(i,row['label'])
#         #循环进行处理
#         if (i + 1) % seq_len == 0:
#             label.append(lv)
#             #重置
#             prev = 0.0
#             lv = [0] * label_len
#     np.save('data.npy',feature)
#     np.save('label.npy',label)
# time_start = time.time()
#取整，最后一点不要了

# split_nums = (length - length % seq_len)/seq_len
# col_size = df['size']
# col_time = df['time']
# col_direct = df['direct']
# col_label = df['label'] + 1
# #转成numpy类型
# col_size = np.array(col_size)[:length - length % seq_len]
# col_label = np.array(col_label)[:length - length % seq_len]
# col_time = np.array(col_time)[:length - length % seq_len]
# col_direct = np.array(col_direct)[:length - length % seq_len]
#
# col_size = np.split(col_size,split_nums)
# col_time = np.split(col_time,split_nums)
# col_direct = np.split(col_direct,split_nums)
# col_label = np.split(col_label,split_nums)
# col_size,col_time,col_direct,col_label = np.array(col_size),np.array(col_time).astype(np.float64),np.array(col_direct).astype(np.int32),np.array(col_label).astype(np.int32)
# labellist = np.zeros((int(split_nums), 26)).astype(np.int32)
#对label内容,以seq_len,进行转换

#相对时间
# split_nums = int(split_nums)
# for i in range(split_nums):
#     pivot = col_time[i][0]
#     for j in range(1,seq_len):
#         if col_time[i][j] - pivot > 0:
#             col_time[i][j] = -np.log10(col_time[i][j] - pivot)
#     col_time[i][0] = 0
# print('11111111111111111111111111')
#对label内容,以seq_len,进行转换
def loadData1():
    seq_len = 20

    df = pd.read_csv('../../fingerprint/data/mixture_10d.csv')
    time = np.array(df['time'])
    label = np.array(df['label'])
    size = np.array(df['size'])
    direct = np.array(df['direct'])
    payload = np.array(df['payload_size'])
    length = len(time)
    remain_len = length - length % seq_len
    split_nums = remain_len / seq_len

    time = time[:remain_len]
    label = label[:remain_len]
    size = size[:remain_len]
    direct = direct[:remain_len]
    payload = payload[:remain_len]

    time = np.split(time, split_nums)
    label = np.split(label, split_nums)
    size = np.split(size, split_nums)
    direct = np.split(direct, split_nums)
    payload = np.split(payload,split_nums)

    time, label, size, direct,payload = np.array(time), np.array(label), np.array(size), np.array(direct),np.array(payload)
    for i in range(time.shape[0]):
        for j in range(1, time.shape[1]):
            if time[i][j] - time[i][0]>0:
                time[i][j] = -np.log10(time[i][j] - time[i][0])
            else:
                time[i][j] = 3.0
            # print(time[i][j])
        time[i][0] = 0
    truelabel = []
    for seq in label:
        each_label = [0]*25
        for i in seq:
            if i>=0:
                each_label[i] = 1
        truelabel.append(each_label)

    direct = np.where(direct == 0, -1, 1)
    truelabel = np.array(truelabel)

    data = np.concatenate([size, time, direct,payload], axis=1)
    data = np.array(data)

    np.save('data1.npy', data)
    np.save('label1.npy', truelabel)

def loadData2():
    seq_len = 20
    data2 = np.load('../data3.npy',allow_pickle=True)
    label2 = np.load('../label3.npy',allow_pickle=True)
    col_time = data2[:,:seq_len]
    for i in range(col_time.shape[0]):
        pivot = col_time[i][0]
        for j in range(1,seq_len):
            if col_time[i][j] - pivot > 0:
                col_time[i][j] = -np.log10(col_time[i][j] - pivot)
        col_time[i][0] = 0

    col_direct = data2[:,30:(30+seq_len)]
    np.where(col_direct == 0,-1,1)
    col_packet_size = data2[:,60:(60+seq_len)]
    col_payload_size = data2[:,90:(90+seq_len)]
    # col_payload_size = data2[:,120:120+seq_len]

    data = np.concatenate([col_packet_size,col_time,col_direct,col_payload_size],axis=1)

    np.save('data2.npy',data)
    np.save('label2.npy',label2)
loadData1()