# DFbaselines说明

**复现论文代码**

使用Tensorflow2复现了三篇论文作为Baseline，分别为 



1.**CNN-RNN**:Deep Learning for Network Traffic Classification
2.**FS-Net**:FS-Net: A Flow Sequence Network For Encrypted Traffic Classification
3.**Multitasks**:Multitask Learning for Network Traffic Classification



**make_traffic.py**：使用scapy提取 30天中25种IoT设备的流量+其他非IoT设备产生的背景流量特征，产生一个按时间排序的的packet序列，以csv形式保存为文件。

然后将此csv文件用两种处理方式进行处理和分割，产生两种数据集



1. 按照**相同长度序列** 对packet序列进行分割（模拟VPN环境）

2. **flow_extract.py**:将具有**相同五元组**的packet归入一个flow（模拟LAN，NAT环境）

   

三种Baseline分别运行在这样的两种数据集上，分别得出在三种网络环境下整体的precison,recall，F1指标以及单个设备的指标，与我们提出的方案进行对比。



