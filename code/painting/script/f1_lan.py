import matplotlib.pyplot as plt
import numpy as np
original=[0.80833099, 0.84532736 ,0.87046866 ,0.90452553, 0.92522445, 0.93238749 ,0.94781127]
sskd=[0.86773809 ,0.90097621, 0.92763494, 0.94311364 ,0.95793542, 0.97719401,0.98288034]
kd = [0.8158992791291554,0.8554116433034848,0.9072135714320785,0.9262990219637656,0.9355737454353021,0.947066575123161,0.9649395322976567]
original,sskd,kd = list(map(lambda x:x*100,original)),list(map(lambda x:x*100,sskd)),list(map(lambda x:x*100,kd))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([79,100.8])
plt.yticks(np.arange(80, 101, 10),fontsize =50)
# plt.xlim([0,6])
plt.plot(x,sskd,label='PS-Tree',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,kd,label='ReDT',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)

plt.plot(x,original,label='DT',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('F1( % )',fontsize = 60)
plt.legend(loc = 'lower right',borderpad=0,fontsize=60)
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/lan_f1.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
