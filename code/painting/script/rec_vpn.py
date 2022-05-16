import matplotlib.pyplot as plt
import numpy as np

original_lan=[0.69872165 ,0.7604282, 0.7764409 ,0.8341664, 0.8707922, 0.8808334,0.9117972 ]
sskd_lan=[0.7690538 , 0.82465607, 0.8794621  ,0.89689034 ,0.9383377  ,0.96757644,0.9764675 ]
kd_lan =[0.73873276,0.7703694,0.8424355,0.8735419,0.89797634,0.9194062,0.9467126]



original_vpn=[0.7168749 , 0.7216869 , 0.7216869 , 0.7229338,  0.72802097,0.7279917,
 0.734677  ]
sskd_vpn=[0.74644953 ,0.7676646,  0.7835115  ,0.790027,   0.8046504 , 0.81545115,
 0.8210008 ]
kd_vpn=[0.71469945, 0.7164847 , 0.7194971 , 0.724097   ,0.7306044 , 0.7333492,
 0.73999035]
original_lan,sskd_lan,kd_lan = list(map(lambda x:x*100,original_lan)),list(map(lambda x:x*100,sskd_lan)),list(map(lambda x:x*100,kd_lan))
original_vpn,sskd_vpn,kd_vpn = list(map(lambda x:x*100,original_vpn)),list(map(lambda x:x*100,sskd_vpn)),list(map(lambda x:x*100,kd_vpn))
fig = plt.figure(dpi=128,figsize=(16, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([50,100])
plt.yticks(np.arange(50, 100, 10),fontsize =50)
# plt.xlim([0,6])
plt.plot(x,sskd_lan,label='PS-Tree-LAN',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,kd_lan,label='ReDT-LAN',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)

plt.plot(x,original_vpn,label='DT-LAN',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
plt.plot(x,sskd_vpn,label='PS-Tree-VPN',linewidth=4,color='c',marker='p',linestyle ='--',markerfacecolor='c',markersize=35)
plt.plot(x,kd_vpn,label='ReDT-VPN',linewidth=4,color='m',linestyle ='--',marker='*',markerfacecolor='m',markersize=35)

plt.plot(x,original_lan,label='DT-VPN',linewidth=4,color='y',linestyle ='--',marker='d',markerfacecolor='y',markersize=35)


# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('Recall( % )',fontsize = 60)
plt.legend(ncol = 2,loc = 'best',borderpad=0,fontsize=50,columnspacing=0.1,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor =(0.88,0.35))
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/rec.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
