import matplotlib.pyplot as plt
import numpy as np
# 95,0.9914333820343018,0.9924179911613464,0.7800747704049367,0.9384438939412802,0.9455057630020984]
mar = [0.9980767369270325,0.9957967400550842,0.9969354349013883]
jun = [0.95331085, 0.92148304, 0.9371267780505829]
sep = [0.9241342, 0.88817954, 0.9058001841522941]
labels = ['Precision','Recall','F1']
mar,jun,sep = list(map(lambda x:x*100,mar)),list(map(lambda x:x*100 ,jun)),list(map(lambda x:x*100 ,sep))
model =['March','June','September']
plt.figure(figsize=(18,9))
plt.rc('font',family='Times New Roman')
plt.figure(1)
plt.ylim([0,8])
plt.ylim([65.0,102.0])
plt.yticks(np.arange(65, 101.2, 5),fontsize = 65)
x = 2*np.arange(len(model))+ 0.7  # x轴刻度标签位置
width = 0.35
plt.xticks(x+0.7 ,model,rotation=0,fontsize = 65)  # 绘制x刻度标签
plt.ylabel('Metrics( % )',fontsize = 65)
plt.xlabel('Timeline',fontsize = 65)
hatch_list=['*','/','o']
ec_list = ['royalblue','tomato','teal']
# ec_list = ['tomato','royalblue','indianred','green','goldenrod','teal','orangered']
inter =0.2
xbar = [0,width,width*2]
for i in range(3):
    plt.bar(1.0+xbar[i], width=width,align='center', hatch =hatch_list[i],height=mar[i],ec=ec_list[i],lw = 2,color = 'w',label = labels[i])

# plt.bar(1.0+width, width=width,align='center', hatch ='x',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[1])
# plt.bar(1.0+width*2, width=width,align='center', hatch ='/',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[2])
# plt.bar(1.0+width*3, width=width,align='center', hatch ='v',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[3])
# plt.bar(1.0+width*4, width=width,align='center', hatch ='',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[4])
# plt.bar(1.0+width*5, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[5])
# plt.bar(1.0+width*6, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[6])
# plt.bar(1.0+width*7, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[7])

for i in range(3):
    plt.bar(3.0+xbar[i], width=width,align='center', hatch =hatch_list[i],height=jun[i],ec=ec_list[i],lw = 2,color = 'w')

# plt.bar(3.0, width=width,align='center', hatch ='*',height=cnn[0],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width, width=width,align='center', hatch ='*',height=cnn[1],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*2, width=width,align='center', hatch ='*',height=cnn[2],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*3, width=width,align='center', hatch ='*',height=cnn[3],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*4, width=width,align='center', hatch ='*',height=cnn[4],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*5, width=width,align='center', hatch ='*',height=cnn[5],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*6, width=width,align='center', hatch ='*',height=cnn[6],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*7, width=width,align='center', hatch ='*',height=cnn[7],ec='tomato',lw = 2,color = 'w')

for i in range(3):
    plt.bar(5.0+xbar[i], width = width, align = 'center', hatch = hatch_list[i], height = sep[i], ec = ec_list[i], lw = 2, color = 'w')


    # plt.bar(5.0, width=width,align='center', hatch ='*',height=lstm[0],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width, width=width,align='center', hatch ='*',height=lstm[1],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*2, width=width,align='center', hatch ='*',height=lstm[2],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*3, width=width,align='center', hatch ='*',height=lstm[3],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*4, width=width,align='center', hatch ='*',height=lstm[4],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*5, width=width,align='center', hatch ='*',height=lstm[5],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*6, width=width,align='center', hatch ='*',height=lstm[6],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*7, width=width,align='center', hatch ='*',height=lstm[7],ec='tomato',lw = 2,color = 'w')

plt.grid(True,linestyle=':',color='b',alpha=0.6,axis='y')
# handles, labels = get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
plt.legend(ncol = 3,loc = 'upper right',borderpad=0,fontsize=55,columnspacing=0.1,labelspacing=0.1,handletextpad=0.1,bbox_to_anchor =(1.035,0.99))
# plt.legend()
plt.savefig("../pic/three_months.pdf",bbox_inches='tight')
# plt.savefig("net_overall.jpg")
plt.show()