import matplotlib.pyplot as plt
import numpy as np
original=[0.76611902 ,0.76996584 ,0.76996584, 0.77341196, 0.77417065, 0.77271572,0.7713308 ]
sskd=   [0.79867711, 0.81193501, 0.82073783, 0.82755275, 0.83493242 ,0.84109945,0.8441654 ]
kd = [0.76550649, 0.76804061 ,0.77102694, 0.77295299, 0.77351074, 0.77197039,0.7701381 ]
original,sskd,kd = list(map(lambda x:x*100,original)),list(map(lambda x:x*100,sskd)),list(map(lambda x:x*100,kd))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([57,91])
plt.yticks(np.arange(60, 91, 10),fontsize =50)
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
plt.savefig('../pic/vpn_f1.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
