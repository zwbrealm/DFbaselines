import matplotlib.pyplot as plt
import numpy as np
# dense =[0.944408655166626,0.9887047410011292,0.6755768060684204,0.9611297845840454,0.3880065083503723,0.8731034994125366]
# cnn =[0.9448923468589783,0.9844623804092407,0.7387322187423706,0.9271174669265747,0.32278212904930115,0.8431508541107178]
# lstm =[0.9725382924079895,0.9914333820343018,0.8817759156227112,0.9391574263572693,0.6994074583053589,0.9377315044403076]
dense = [0.4929,0.9150044169752525,0.93783456]
cnn = [0.449262996508783,0.8831428262152574,0.90726855]
lstm = [0.780074799810811,0.9384439237435845,0.95834567]

labels = ['F1_w/o','F1_new','F1_Pac2Vec']
dense,cnn,lstm = list(map(lambda x:x*100,dense)),list(map(lambda x:x*100 ,cnn)),list(map(lambda x:x*100 ,lstm))
model =['MLP','CNN','LSTM']
plt.figure(figsize=(18,9))
plt.rc('font',family='Times New Roman')
plt.figure(1)
plt.ylim([30.0,110.0])
plt.yticks(np.arange(30, 105, 20),fontsize = 65)
x = np.arange(len(model))+ 1  # x轴刻度标签位置
width = 0.25
plt.xticks(x+0.27 ,model,rotation=0,fontsize = 65)  # 绘制x刻度标签
plt.ylabel('Metrics( % )',fontsize = 65)
plt.xlabel('Classifier',fontsize = 65)
hatch_list=['*','+','/','x','O','o']
ec_list = ['tomato','royalblue','indianred','green','goldenrod','teal','orangered']
inter =0.2
xbar = [0,width,width*2]
for i in range(3):
    plt.bar(1.0+xbar[i], width=width,align='center', hatch =hatch_list[i],height=dense[i],ec=ec_list[i],lw = 2,color = 'w',label = labels[i])

# plt.bar(1.0+width, width=width,align='center', hatch ='x',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[1])
# plt.bar(1.0+width*2, width=width,align='center', hatch ='/',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[2])
# plt.bar(1.0+width*3, width=width,align='center', hatch ='v',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[3])
# plt.bar(1.0+width*4, width=width,align='center', hatch ='',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[4])
# plt.bar(1.0+width*5, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[5])
# plt.bar(1.0+width*6, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[6])
# plt.bar(1.0+width*7, width=width,align='center', hatch ='*',height=dense[0],ec='tomato',lw = 2,color = 'w',label = labels[7])

for i in range(3):
    plt.bar(2.0+xbar[i], width=width,align='center', hatch =hatch_list[i],height=cnn[i],ec=ec_list[i],lw = 2,color = 'w')

# plt.bar(3.0, width=width,align='center', hatch ='*',height=cnn[0],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width, width=width,align='center', hatch ='*',height=cnn[1],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*2, width=width,align='center', hatch ='*',height=cnn[2],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*3, width=width,align='center', hatch ='*',height=cnn[3],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*4, width=width,align='center', hatch ='*',height=cnn[4],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*5, width=width,align='center', hatch ='*',height=cnn[5],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*6, width=width,align='center', hatch ='*',height=cnn[6],ec='tomato',lw = 2,color = 'w')
# plt.bar(3.0+width*7, width=width,align='center', hatch ='*',height=cnn[7],ec='tomato',lw = 2,color = 'w')

for i in range(3):
    plt.bar(3.0+xbar[i], width = width, align = 'center', hatch = hatch_list[i], height = lstm[i], ec = ec_list[i], lw = 2, color = 'w')


    # plt.bar(5.0, width=width,align='center', hatch ='*',height=lstm[0],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width, width=width,align='center', hatch ='*',height=lstm[1],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*2, width=width,align='center', hatch ='*',height=lstm[2],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*3, width=width,align='center', hatch ='*',height=lstm[3],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*4, width=width,align='center', hatch ='*',height=lstm[4],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*5, width=width,align='center', hatch ='*',height=lstm[5],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*6, width=width,align='center', hatch ='*',height=lstm[6],ec='tomato',lw = 2,color = 'w')
# plt.bar(5.0+width*7, width=width,align='center', hatch ='*',height=lstm[7],ec='tomato',lw = 2,color = 'w')

plt.grid(True,linestyle=':',color='b',alpha=0.6,axis='y')
# bbox_to_anchor =(0.953,0.98)
plt.legend(ncol=3,borderpad=0,fontsize=55,bbox_to_anchor =(1.01,1.01),columnspacing=0.1,labelspacing=0.1,handletextpad=0.1)
# plt.legend()
plt.savefig("../pic/distance_matrix.pdf",bbox_inches='tight')
# plt.savefig("net_overall.jpg")
plt.show()