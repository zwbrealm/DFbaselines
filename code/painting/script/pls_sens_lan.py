import matplotlib.pyplot as plt
import numpy as np
pre = [0.94067734 ,0.96266162, 0.98117185, 0.99065185, 0.99641591, 0.99848521]
rec = [0.94118756 ,0.96088338 ,0.97444528 ,0.97683209 ,0.9950071 , 0.9950186 ]
f1 =  [0.94093238, 0.96177168, 0.97779697, 0.9836934 , 0.99571101, 0.99674889]
pre,rec,f1 = list(map(lambda x:x*100,pre)),list(map(lambda x:x*100,rec)),list(map(lambda x:x*100,f1))
fig = plt.figure(dpi=128,figsize=(18, 9))
plt.rc('font',family='Times New Roman')
x = [1,2,3,4,5,6]
plt.xticks(x, [5,10,20,50,100,200],fontsize = 65)
plt.ylim([90,103])
plt.yticks(np.arange(90, 100.5, 2),fontsize = 65)
# plt.xlim([0,6])
plt.plot(x,pre,label='Precision',linewidth=4,color='g',marker='s',linestyle ='--',markerfacecolor='g',markersize=35)
plt.plot(x,rec,label='Recall',linewidth=4,color='r',linestyle ='--',marker='o',markerfacecolor='r',markersize=35)
plt.plot(x,f1,label='F1',linewidth=4,color='b',linestyle ='--',marker='v',markerfacecolor='b',markersize=35)
# plt.xscale("log")
# plt.xscale("log")
plt.xlabel('Length of packet sequence',fontsize = 65)
plt.ylabel('Metrics( % )',fontsize = 65)

# plt.title('Interesting Graph\nCheck it out')
plt.grid(True,linestyle=':',color='b',alpha=0.6)

plt.legend(loc = 'lower right',borderpad=0,fontsize=55,bbox_to_anchor =(1.02,-0.06))
# bbox_to_anchor =(2.02,-0.02)
# review/pic/
plt.savefig("../pic/sensitivity_exp.pdf",bbox_inches='tight')
# plt.savefig("tree_pre.jpg")
plt.show()
