import pandas as pd
import numpy as np
import time as tm
# 前n个包
flow_seq_length = 30
hashlist = dict()
class flow:
    def __init__(self,five_tuple):
        self.five_tuple = five_tuple
        self.time_seq = []
        self.packet_size = []
        self.payload_size = []
        self.label = -1
        self.direct = []
        self.length = 0

def flow_exist(flows,i):
    #len(words) - index - 1
    if i.five_tuple in hashlist.keys():

        # hashlist.reverse()
        #得到最后出现某个值的下标
        # k = len(hashlist) - k -1
        # hashlist.reverse()
        # 长度大于限制，跳过
        n = hashlist[i.five_tuple]
        if flows[n].length < flow_seq_length:
            flows[n].time_seq.extend(i.time_seq)
            flows[n].packet_size.extend(i.packet_size)
            flows[n].payload_size.extend(i.payload_size)
            flows[n].direct.extend(i.direct)
            flows[n].length += 1
            return True
    else:
        flows.append(i)
        hashlist[i.five_tuple] =len(flows)-1
def make_label(label):
    if label == -1:
        return [0]*25
    else:
        lv = [0]*25
        lv[label] = 1
        return lv

df = pd.read_csv('../fingerprint/data/mixture_10d.csv',low_memory=False)
flows = []
#数值类型转换
df['payload_size'] = df['payload_size'].astype(np.int32)
df['size'] = df['size'].astype(np.int32)
df['direct'] = df['direct'].astype(np.int32)
df['label'] = df['label'].astype(np.int32)
length = df.shape[0]
for i,row in df.iterrows():
    time = row['time']
    payload_size = row['payload_size']
    packet_size = row['size']
    direct = row['direct']
    label = row['label']
    five_tuple =(row['src'],row['sport'],row['dest'],row['dport'],row['proto'])

    new_f = flow(five_tuple)
    new_f.time_seq.append(time)
    new_f.payload_size.append(payload_size)
    new_f.packet_size.append(packet_size)
    new_f.label = label
    new_f.length = 1
    new_f.direct.append(direct)
    # t1 = tm.time()
    flow_exist(flows,new_f)
    if i%5000 == 0:
        print('%s'%(str(float(i)/length)))

print('-----------------------------------------')
datas = []
labels = []
for i in flows:
    item = []
    # print(i.time_seq)
    # print(i.direct)
    # print(i.packet_size)
    # print(i.payload_size)
    # print('-------------------')

    length_i = len(i.time_seq)
    #差多少，补全
    if length_i < flow_seq_length:
        i.time_seq.extend([0]*(flow_seq_length-length_i))
        i.direct.extend([0]*(flow_seq_length-length_i))
        i.packet_size.extend([0]*(flow_seq_length-length_i))
        i.payload_size.extend([0]*(flow_seq_length-length_i))

    item.extend(i.time_seq)
    item.extend(i.direct)
    item.extend(i.packet_size)
    item.extend(i.payload_size)
    # print(item,len(item))
    l = make_label(i.label)
    datas.append(item)
    labels.append(l)
datas = np.array(datas)
labels = np.array(labels)
np.save('data3.npy',datas,allow_pickle=True)
np.save('label3.npy',labels,allow_pickle=True)