import matplotlib.pyplot as plt
import numpy as np
pre = [1.       ,  0.9985543 , 0.996511  , 0.99641591, 0.9961365 ]
rec = [0.99921817, 0.9981334 , 0.99442923 ,0.9950071,  0.99293965]
f1 = [0.9996089  ,0.99834384, 0.99546902, 0.99571101, 0.9945355 ]
acc,pre,rec = list(map(lambda x:x*100,pre)),list(map(lambda x:x*100,rec)),list(map(lambda x:x*100,f1))
fig = plt.figure(dpi=128,figsize=(16, 7))
plt.rc('font',family='Times New Roman')
x = [1,2,3,4,5]
plt.xticks(x, [5,10,15,20,25],fontsize = 40)
plt.ylim([60,101.9])
plt.yticks(np.arange(60, 100.1, 10),fontsize = 43)
# plt.xlim([0,6])
plt.plot(x,acc,label='Precision',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,pre,label='Recall',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)
plt.plot(x,rec,label='F1',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
plt.xlabel('Decice number',fontsize = 40)
plt.ylabel('Metrics( % )',fontsize = 43)

# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)

plt.legend(loc = 'lower right',borderpad=0,fontsize=50,bbox_to_anchor =(1.02,0.01))
# bbox_to_anchor =(2.02,-0.02)
# review/pic/
plt.savefig("../pic/dev_num_lan.pdf",bbox_inches='tight')
# plt.savefig("tree_pre.jpg")
plt.show()
