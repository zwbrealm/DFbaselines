import matplotlib.pyplot as plt
import numpy as np

f1 = [0.7947925329756994,0.9569745702019724,0.8610262475024411,0.31635748668488206,0.5162798678900498,0.4947628196096621,0.49787177156902507]
rec = [0.7547810673713684,0.9607115387916564,0.7845937013626099,0.4800349040139616,0.3808578,0.371743,0.3866428443061011]
pre = [0.8392834663391113,0.9532665610313416,0.9539576768875122,0.9278166666666666,0.80114335,0.73947513,0.6989427401701961]
plt.figure(figsize=(16,8))
f1 = [i*100 for i in f1]
rec = [i*100 for i in rec]
pre= [i*100 for i in pre]
plt.rc('font',family='Times New Roman')
plt.figure(1)
plt.ylim([30,120.0])
plt.yticks(np.arange(30, 100.2, 10),fontsize = 40)
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
plt.savefig("../pic/overall_vpn.pdf",bbox_inches='tight')
# plt.savefig("net_overall.jpg")
plt.show()