import matplotlib.pyplot as plt
import numpy as np

pre_lan = [1.       ,  0.9985543 , 0.996511  , 0.99641591, 0.9961365 ]
rec_lan = [0.99921817, 0.9981334 , 0.99442923 ,0.9950071,  0.99293965]
f1_lan = [0.9996089  ,0.99834384, 0.99546902, 0.99571101, 0.9945355 ]

pre_vpn = [0.9960537 , 0.9959111 , 0.9798414 , 0.97603935, 0.97313643]
rec_vpn = [0.99666303, 0.99557376, 0.9786316,  0.97559166 ,0.97599429]
f1_vpn = [0.99635824 ,0.99574241, 0.97923617, 0.97581542, 0.97456323]
pre_lan,rec_lan,f1_lan = list(map(lambda x:x*100,pre_lan)),list(map(lambda x:x*100,rec_lan)),list(map(lambda x:x*100,f1_lan))
pre_vpn,rec_vpn,f1_vpn = list(map(lambda x:x*100,pre_vpn)),list(map(lambda x:x*100,rec_vpn)),list(map(lambda x:x*100,f1_vpn))
fig = plt.figure(dpi=128,figsize=(18, 9))
plt.rc('font',family='Times New Roman')
x = [1,2,3,4,5]
plt.xticks(x, [5,10,15,20,25],fontsize = 65)
plt.ylim([90,102.9])
plt.yticks(np.arange(90, 100.1, 5),fontsize = 65)
# plt.xlim([0,6])
plt.plot(x,pre_lan,label='Precision-LAN',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,f1_lan,label='F1-LAN',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)

plt.plot(x,rec_lan,label='Recall-LAN',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)
plt.plot(x,pre_vpn,label='Precision-VPN',linewidth=4,color='c',marker='p',linestyle ='--',markerfacecolor='c',markersize=35)
plt.plot(x,f1_vpn,label='F1-VPN',linewidth=4,color='y',linestyle ='--',marker='^',markerfacecolor='y',markersize=35)
plt.plot(x,rec_vpn,label='Recall-VPN',linewidth=4,color='m',linestyle ='--',marker='*',markerfacecolor='m',markersize=35)


# plt.xscale("log")
# plt.xscale("log")
plt.xlabel('Number of device types',fontsize = 65)
plt.ylabel('Metrics( % )',fontsize = 65)

# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)

plt.legend(ncol = 2,loc = 'best',borderpad=0,fontsize=55,columnspacing=0.1,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor =(0.96,0.39))
# bbox_to_anchor =(2.02,-0.02)
# review/pic/
plt.savefig("../pic/dev_num.pdf",bbox_inches='tight')
# plt.savefig("tree_pre.jpg")
plt.show()