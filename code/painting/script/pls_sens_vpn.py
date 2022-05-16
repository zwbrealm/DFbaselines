import matplotlib.pyplot as plt
import numpy as np

pre = [0.90441465, 0.93183625, 0.94616306, 0.95519453, 0.97313643, 0.97899091]
rec = [0.90556329, 0.92120606, 0.94642192, 0.97477347, 0.97599429, 0.98544943]
f1 = [0.90498858 ,0.92649063 ,0.94629244, 0.96488469, 0.97456323, 0.98220956]
acc,pre,rec = list(map(lambda x:x*100,pre)),list(map(lambda x:x*100,rec)),list(map(lambda x:x*100,f1))
fig = plt.figure(dpi=128,figsize=(16, 7))
plt.rc('font',family='Times New Roman')
x = [1,2,3,4,5,6]
plt.xticks(x, [5,10,20,50,100,200],fontsize = 40)
plt.ylim([90,101.8])
plt.yticks(np.arange(90, 100.1, 2),fontsize = 43)
# plt.xlim([0,6])
plt.plot(x,acc,label='Precision',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,pre,label='Recall',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)
plt.plot(x,rec,label='F1',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
plt.xlabel('Packet sequence length',fontsize = 40)
plt.ylabel('Metrics( % )',fontsize = 43)

# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)

plt.legend(loc = 'lower right',borderpad=0,fontsize=50,bbox_to_anchor =(1.02,-0.06))
# bbox_to_anchor =(2.02,-0.02)
plt.savefig("../pic/seq_sensitivity_vpn.pdf",bbox_inches='tight')
# plt.savefig("tree_pre.jpg")
plt.show()
