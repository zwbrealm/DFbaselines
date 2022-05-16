import matplotlib.pyplot as plt
import numpy as np
original=[0.82262766, 0.82516736, 0.82516736 ,0.83146834, 0.82656723, 0.8232946,0.8118341 ]
sskd=[0.8587631  ,0.861624 ,  0.861678 ,  0.8688212 , 0.86758274, 0.8684135,0.8686752 ]
kd=[0.8240899  ,0.8275913 , 0.83050734, 0.82887876, 0.821771 ,  0.8148857,0.8028467 ]
original,sskd,kd = list(map(lambda x:x*100,original)),list(map(lambda x:x*100,sskd)),list(map(lambda x:x*100,kd))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([65,90.5])
plt.yticks(np.arange(65, 91, 5),fontsize =50)
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
plt.legend(loc = 'lower right',borderpad=0,fontsize=60,bbox_to_anchor =(1.02,0.01))
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/vpn_pre.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
