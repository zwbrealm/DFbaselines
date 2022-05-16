import matplotlib.pyplot as plt
import numpy as np



original_lan=[0.95872784 ,0.95156646, 0.99040794 ,0.98784703, 0.98691535, 0.9903516,0.9867873 ]
sskd_lan=[0.99547666 ,0.9928636 , 0.981391   ,0.99436027, 0.9783692 , 0.98700476,0.9893779 ]
kd_lan =[0.91106755,0.96155965,0.9827835,0.98583823,0.976457,0.976443,0.98388207]

original_vpn=[0.82262766, 0.82516736, 0.82516736 ,0.83146834, 0.82656723, 0.8232946,0.8118341 ]
sskd_vpn=[0.8587631  ,0.861624 ,  0.861678 ,  0.8688212 , 0.86758274, 0.8684135,0.8686752 ]
kd_vpn=[0.8240899  ,0.8275913 , 0.83050734, 0.82887876, 0.821771 ,  0.8148857,0.8028467 ]
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



# plt.xscale("log")
# plt.xscale("log")
# plt.plot(x,kd,label='kd',linewidth=3,color='chartreuse',marker='o',linestyle='--',markerfacecolor='chartreuse',markersize=16)

# plt.xscale("log")


plt.xlabel('Depth',fontsize = 60)
plt.ylabel('Precision( % )',fontsize = 60)
plt.legend(ncol = 2,loc = 'best',borderpad=0,fontsize=50,columnspacing=0.1,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor =(0.96,0.35))
# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.savefig('../pic/prec.pdf',bbox_inches='tight')
# plt.savefig("tree_acc.svg")

plt.show()
