import matplotlib.pyplot as plt
import numpy as np
original=[0.69872165 ,0.7604282, 0.7764409 ,0.8341664, 0.8707922, 0.8808334,0.9117972 ]
sskd=[0.7690538 , 0.82465607, 0.8794621  ,0.89689034 ,0.9383377  ,0.96757644,0.9764675 ]
kd =[0.73873276,0.7703694,0.8424355,0.8735419,0.89797634,0.9194062,0.9467126]
original,sskd,kd = list(map(lambda x:x*100,original)),list(map(lambda x:x*100,sskd)),list(map(lambda x:x*100,kd))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([68,100.8])
plt.yticks(np.arange(65, 101, 10),fontsize =50)
# plt.xlim([0,6])
plt.plot(x,sskd,label='PS-Tree',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,kd,label='ReDT',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)

plt.plot(x,original,label='DT',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('Recall( % )',fontsize = 60)
plt.legend(loc = 'lower right',borderpad=0,fontsize=60)
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/lan_rec.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
