import matplotlib.pyplot as plt
import numpy as np

f1 = [0.7947925329756994,0.9955951007299167,0.9647069277259895,0.3243050377862122,0.8148995039159509,0.8434660310929898,0.8844907694090105]
rec = [0.7547810673713684,0.9936814904212952,0.9474622011184692,0.48,0.7527761,0.83539134,0.8664730699687688]
pre = [0.8392834663391113,0.9975160956382751,0.9825910329818726,0.9998166666666667,0.88819885,0.85169834,0.9032737123833149]
plt.figure(figsize=(16,8))
f1 = [i*100 for i in f1]
rec = [i*100 for i in rec]
pre= [i*100 for i in pre]
plt.rc('font',family='Times New Roman')
plt.figure(1)
plt.ylim([30,120.0])
plt.yticks(np.arange(30, 100.2, 20),fontsize = 40)
labels = ['PS-Net', 'HomeMole', 'DarkSide','Pinpoint','CNN-RNN' ,'Multitask','FS-Net']
labels = [labels[0],labels[2],labels[1],labels[3],labels[4],labels[5],labels[6]]
x = np.arange(len(labels))+1.25  # x轴刻度标签位置
width = 0.25
plt.xticks(x, labels,rotation=25,fontsize = 45)  # 绘制x刻度标签
plt.ylabel('Metrics( % )',fontsize = 45)
# plt.bar(0, 'PS-Net', hatch = ['-' , '+' , 'x ', 'o','/','.','*'])

plt.bar(1.0, width=width,align='center', hatch ='x',height=pre[1],ec='royalblue',lw = 2,color = 'w',label = 'Precision')
plt.bar(1.0+width, width=width,align='center', hatch ='o',height=rec[1],ec='goldenrod',lw = 2,color = 'w',label = 'Recall')
plt.bar(1.0+width*2, width=width,align='center', hatch ='*',height=f1[1],ec='tomato',lw = 2,color = 'w',label = 'F1')
plt.bar(2.0+width*2, width=width,align='center', hatch ='*',height=f1[0],ec='tomato',lw = 2,color = 'w')
plt.bar(2.0, width=width,align='center', hatch ='x',height=pre[0],ec='royalblue',lw = 2,color = 'w')
plt.bar(2.0+width, width=width,align='center', hatch ='o',height=rec[0],ec='goldenrod',lw = 2,color = 'w')
plt.bar(3.0+width*2, width=width,align='center', hatch ='*',height=f1[2],ec='tomato',lw = 2,color = 'w')
plt.bar(3.0, width=width,align='center', hatch ='x',height=pre[2],ec='royalblue',lw = 2,color = 'w')
plt.bar(3.0+width, width=width,align='center', hatch ='o',height=rec[2],ec='goldenrod',lw = 2,color = 'w')
plt.bar(4.0+width*2, width=width,align='center', hatch ='*',height=f1[3],ec='tomato',lw = 2,color = 'w')
plt.bar(4.0, width=width,align='center', hatch ='x',height=pre[3],ec='royalblue',lw = 2,color = 'w')
plt.bar(4.0+width, width=width,align='center', hatch ='o',height=rec[3],ec='goldenrod',lw = 2,color = 'w')
plt.bar(5.0+width*2, width=width,align='center', hatch ='*',height=f1[4],ec='tomato',lw = 2,color = 'w')
plt.bar(5.0, width=width,align='center', hatch ='x',height=pre[4],ec='royalblue',lw = 2,color = 'w')
plt.bar(5.0+width, width=width,align='center', hatch ='o',height=rec[4],ec='goldenrod',lw = 2,color = 'w')
plt.bar(6.0+width*2, width=width,align='center', hatch ='*',height=f1[5],ec='tomato',lw = 2,color = 'w')
plt.bar(6.0, width=width,align='center', hatch ='x',height=pre[5],ec='royalblue',lw = 2,color = 'w')
plt.bar(6.0+width, width=width,align='center', hatch ='o',height=rec[5],ec='goldenrod',lw = 2,color = 'w')
plt.bar(7.0+width*2, width=width,align='center', hatch ='*',height=f1[6],ec='tomato',lw = 2,color = 'w')
plt.bar(7.0, width=width,align='center', hatch ='x',height=pre[6],ec='royalblue',lw = 2,color = 'w')
plt.bar(7.0+width, width=width,align='center', hatch ='o',height=rec[6],ec='goldenrod',lw = 2,color = 'w')
plt.grid(True,linestyle=':',color='b',alpha=0.6)
plt.legend(ncol=3,loc = 'best',borderpad=0,fontsize=44,bbox_to_anchor =(0.89,1.00),columnspacing=0.2)
plt.savefig("../pic/overall_nat.pdf",bbox_inches='tight')
# plt.savefig("net_overall.jpg")
plt.show()