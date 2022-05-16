import matplotlib.pyplot as plt
import numpy as np
original=[0.95872784 ,0.95156646, 0.99040794 ,0.98784703, 0.98691535, 0.9903516,0.9867873 ]
sskd=[0.99547666 ,0.9928636 , 0.981391   ,0.99436027, 0.9783692 , 0.98700476,0.9893779 ]
kd =[0.91106755,0.96155965,0.9827835,0.98583823,0.976457,0.976443,0.98388207]
original,sskd,kd = list(map(lambda x:x*100,original)),list(map(lambda x:x*100,sskd)),list(map(lambda x:x*100,kd))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([89,100.8])
plt.yticks(np.arange(90, 101, 5),fontsize =50)
# plt.xlim([0,6])
plt.plot(x,sskd,label='PS-Tree',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,kd,label='ReDT',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)

plt.plot(x,original,label='DT',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('Precision( % )',fontsize = 60)
plt.legend(loc = 'lower right',borderpad=0,fontsize=60)
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/lan_pre.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
