import matplotlib.pyplot as plt
import numpy as np



original_lan=[0.80833099, 0.84532736 ,0.87046866 ,0.90452553, 0.92522445, 0.93238749 ,0.94781127]
sskd_lan=[0.86773809 ,0.90097621, 0.92763494, 0.94311364 ,0.95793542, 0.97719401,0.98288034]
kd_lan = [0.8158992791291554,0.8554116433034848,0.9072135714320785,0.9262990219637656,0.9355737454353021,0.947066575123161,0.9649395322976567]


original_vpn=[0.76611902 ,0.76996584 ,0.76996584, 0.77341196, 0.77417065, 0.77271572,0.7713308 ]
sskd_vpn= [0.79867711, 0.81193501, 0.82073783, 0.82755275, 0.83493242 ,0.84109945,0.8441654 ]
kd_vpn = [0.76550649, 0.76804061 ,0.77102694, 0.77295299, 0.77351074, 0.77197039,0.7701381 ]
original_lan,sskd_lan,kd_lan = list(map(lambda x:x*100,original_lan)),list(map(lambda x:x*100,sskd_lan)),list(map(lambda x:x*100,kd_lan))
original_vpn,sskd_vpn,kd_vpn = list(map(lambda x:x*100,original_vpn)),list(map(lambda x:x*100,sskd_vpn)),list(map(lambda x:x*100,kd_vpn))
fig = plt.figure(dpi=128,figsize=(21, 9))
plt.rc('font',family='Times New Roman')
x = [6,7,8,9,10,11,12]
# labels = ['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$']
plt.xticks(x,fontsize = 50)
plt.ylim([50,105])
plt.yticks(np.arange(50, 105, 10),fontsize =50)
# plt.xlim([0,6])
plt.plot(x,sskd_lan,label='PS-Tree-LAN',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,original_lan,label='DT-LAN',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)

plt.plot(x,kd_lan,label='ReDT-LAN',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)

plt.plot(x,sskd_vpn,label='PS-Tree-VPN',linewidth=4,color='c',marker='p',linestyle ='--',markerfacecolor='c',markersize=35)
plt.plot(x,original_vpn,label='DT-VPN',linewidth=4,color='y',linestyle ='--',marker='^',markerfacecolor='y',markersize=35)
plt.plot(x,kd_vpn,label='ReDT-VPN',linewidth=4,color='m',linestyle ='--',marker='*',markerfacecolor='m',markersize=35)

# plt.plot(x,original_vpn,label='DT-VPN',linewidth=4,color='y',linestyle ='--',marker='^',markerfacecolor='y',markersize=35)


# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('F1( % )',fontsize = 60)
plt.legend(ncol = 2,loc = 'best',borderpad=0,fontsize=50,columnspacing=0.1,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor =(0.96,0.35))
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/f1.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
