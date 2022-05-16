import pandas as pd
from scapy.all import *
import copy
import time

dest = []
time, size, direct, label = [], [], [], []
sport, dport, proto = [], [], []
ttl, flag, window, options = [], [], [], []


# with PcapReader('../Monday-WorkingHours.pcap') as ps:
with PcapReader('../2017-04-19_win-normal.pcap') as ps:
    for p in ps:
    # ignore other protocols other than TCP and UDP
    #     if 'IP' not in p or (TCP not in p and UDP not in p):
        if not p.haslayer("IP"):
            continue
        if (p['IP'].proto != 6):
            continue
        # ignore LAN traffic
        if p['IP'].src.startswith('10.') and p['IP'].dst.startswith('10.'):
            continue
        # ignore broadcasting traffic
        if p['IP'].dst.endswith('.255'):
            continue
        # ignore multicasting traffic
        if '224.0.0.0' <= p['IP'].src <= '239.255.255.255' or '224.0.0.0' <= p['IP'].dst <= '239.255.255.255':
            continue
        # for device labeling; not necessary for background traffic
        found_flag = 0
        # for i, dev in enumerate(dev_list):
        #     if index_by == 'mac':
        #         if p[Ether].src == dev_dict[dev] or p[Ether].dst == dev_dict[dev]:
        #             label.append(i)
        #             found_flag = 1
        #             break
        #     else:
        #         if p['IP'].src == dev_dict[dev] or p['IP'].dst == dev_dict[dev]:
        #             label.append(i)
        #             found_flag = 1
        #             break
        # if found_flag == 0:
        #     continue

        # for background traffic
        #         label.append(-1)
        label.append(-1)
        time.append(p.time)
        size.append(min(p['IP'].len, 1500))
        d = 0 if p['IP'].src.startswith('192.168.') else 1
        ip_dest = p['IP'].dst if p['IP'].src.startswith('192.168.') else p['IP'].src
        dest.append(ip_dest)
        direct.append(d)
        sport.append(p.sport)
        dport.append(p.dport)
        proto.append(p.proto)
        ttl.append(p.ttl)
        if p.payload.proto == 6:
            flag.append(int(p.payload.payload.flags))
            window.append(p.payload.window)
            ops = ''
            try:
                if len(p.payload.options) == 0:
                    ops = 'null'
                else:
                    for op in p.payload.options:
                        ops = ops + op[0]
                        ops = ops + ','
                options.append(ops)
            except:
                options.append('null')
        else:
            flag.append(0)
            window.append(0)
            options.append('null')
        #     print(len(window), len(options), len(label))
dframe =pd.DataFrame({'dest': dest,
                 'time': time,
                 'size': size,
                 'direct': direct,
                 'sport': sport,
                 'dport': dport,
                 'proto': proto,
                 'label': label,
                 'ttl': ttl,
                 'flag': flag,
                 'window': window,
                 'options': options
                })
dframe.to_csv('new_background_traffic.csv',index = False)
print("handle background traffic to csv ended!")
hour_8_1 = pd.read_csv('background_traffic.csv')
# #时间平移
length = hour_8_1.shape[0]
for i in range(length):
    hour_8_1.loc[i,'time'] += 131342648.0
hour_8_2 = copy.deepcopy(hour_8_1)
hour_8_3 = copy.deepcopy(hour_8_1)
for i in range(length):
    hour_8_2.loc[i,'time'] += 480
for i in range(length):
    hour_8_2.loc[i,'time'] += 960
#
hour_8_12 = pd.concat([hour_8_1,hour_8_2])
hour_8_123 = pd.concat([hour_8_12,hour_8_3])
hour_8_123.to_csv('24h.csv',index = False)

print("24h sample created!")

origin = pd.read_csv('25dev_1mo_raw.csv')
each_day = pd.read_csv('24h.csv')

length = each_day.shape[0]
#30天
for i in range(1,30):
    each_day_tmp = copy.deepcopy(each_day)
    for j in range(length):
        each_day_tmp.loc[j,'time'] += i*86400
    if i > 1:
        each_day = pd.concat([each_day, each_day_tmp])
mixtrue = pd.concat([origin,each_day])
mixtrue =  mixtrue.sort_values(by =['time'],ascending=True)
mixtrue.to_csv('30_days.csv',index = False)
