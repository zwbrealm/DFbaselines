import random
from tensorflow.python.keras.optimizer_v2.adam import Adam
# from keras.optimizer_v2.adam import Adam
from utils1 import precision, recall, f1, bi_acc, acc_each, precision_each, recall_each, f1_each, acc_combine, \
    precision_combine, recall_combine, f1_combine
import numpy as np
import os
import tensorflow as tf
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# # 定义TensorFlow配置
# config = ConfigProto()
# # 配置GPU内存分配方式，按需增长，很关键
# config.gpu_options.allow_growth = True
# # 在创建session的时候把config作为参数传进去
# session = InteractiveSession(config=config)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

timestep = 25
np.random.seed(42)

num_class = 25
train_sample_per_class = 3000
lambda_value = 1

trainData = np.load("trainData1.npy", allow_pickle=True)
trainlabel = np.load("trainLabel1.npy", allow_pickle=True)
trainlabel_ = np.load("trainLabel_1.npy", allow_pickle=True)
print('trainData:', trainData.shape, trainlabel.shape)
# trainData = trainData[:, -timestep*2:]
# trainlabel = trainlabel[:, -timestep*2:]


# trainData = trainData[:timestep*2]
# trainlabel = trainlabel[:timestep*2]
# print(trainData.shape,trainlabel.shape)
mask_d = 256
trainmask = np.zeros((trainlabel.shape[0], mask_d))

class_counter = np.zeros((num_class))
train_size = trainlabel.shape[0]
j = 0

#
# map = np.arange(0,mask_d)
# map = map.reshape((num_class,16))

# maskindex = np.random.randint(0,train_size,train_sample_per_class*20).tolist()
# line_sum = np.ones((25))
# for i in range (0,train_size):
#     # line_sum[np.sum(trainlabel_[i,])]+=1
#     if np.sum(trainlabel_[i,])>7:
#         trainmask[i,:] = 1
# print(line_sum)
# 对train,val,test的不同处理
# train的mask策略：是对每一class中只将前40（train_sample_per_class）的样本条数mask为1
# val和test的mask是（条数，256）的全1 np矩阵
for i in range(train_size):
    class_id = [i for i, x in enumerate(trainlabel_[i]) if x == 1]
    for a in class_id:
        if class_counter[a] < train_sample_per_class:
            # for k in map[a]:
            trainmask[i, :] = 1
        class_counter[a] += 1
# print("unmasked samples: ", str(np.sum(trainmask==1)/256))


# valData = np.load("valData2.npy", allow_pickle=True)
# valLabel = np.load("valLabel2.npy", allow_pickle=True)
# valLabel_ = np.load("valLabel_2.npy", allow_pickle=True)
# print('valData:', valData.shape, valLabel.shape)

# testData = testData[:, -timestep*2:]
# testLabel = testLabel[:, -timestep*2:]

# valData = valData[:, :timestep*2]
# valLabel = valLabel[:timestep*2]
# print(valData.shape,valLabel.shape)

# valLabel = valLabel.astype(int)
# valmask = np.zeros((valLabel.shape[0], mask_d))
# valmask[:, :] = 1

testData = np.load("testData1.npy", allow_pickle=True)
testLabel = np.load("testLabel1.npy", allow_pickle=True)
testLabel_ = np.load("testLabel_1.npy", allow_pickle=True)
# print('testData:',testData.shape,testLabel.shape)
# testData = testData[:, :timestep*2]
# testLabel = testLabel[:timestep*2]
# print(testData.shape,testLabel.shape)
testmask = np.ones((testLabel.shape[0], mask_d))
testmask[:, :] = 1

# 对Bandwidth和Duration进行了量化，每一个范围用一个具体的值代替
for i in range(trainlabel.shape[0]):
    # Categorizing Bandwidth
    if trainlabel[i, 0] < 1500:
        trainlabel[i, 0] = 1
    elif trainlabel[i, 0] < 4500:
        trainlabel[i, 0] = 2
    elif trainlabel[i, 0] < 7500:
        trainlabel[i, 0] = 3
    elif trainlabel[i, 0] < 10000:
        trainlabel[i, 0] = 4
    else:
        trainlabel[i, 0] = 5
    # Categorizing Duration
    if trainlabel[i, 1] < 1:
        trainlabel[i, 1] = 1
    elif trainlabel[i, 1] < 2.5:
        trainlabel[i, 1] = 2
    elif trainlabel[i, 1] < 3.5:
        trainlabel[i, 1] = 3
    else:
        trainlabel[i, 1] = 4


for i in range(testLabel.shape[0]):
    # Categorizing Bandwidth
    if testLabel[i, 0] < 1500:
        testLabel[i, 0] = 1
    elif testLabel[i, 0] < 4500:
        testLabel[i, 0] = 2
    elif testLabel[i, 0] < 7500:
        testLabel[i, 0] = 3
    elif testLabel[i, 0] < 10000:
        testLabel[i, 0] = 4
    else:
        testLabel[i, 0] = 5
    # Categorizing Duration
    if testLabel[i, 1] < 1.5:
        testLabel[i, 1] = 1
    elif testLabel[i, 1] < 2.5:
        testLabel[i, 1] = 2
    elif testLabel[i, 1] < 3.5:
        testLabel[i, 1] = 3
    else:
        testLabel[i, 1] = 4
# label中，Bandwidth有5种取值，Duration有4种，class有4种
# 下面是进行one-hot编码
# 造出三个矩阵，将label的三种特征用one-hot 编码嵌入矩阵中
train_size = trainlabel.shape[0]
Y_train1 = np.zeros((train_size, 5))

Y_train1[np.arange(train_size), (trainlabel[:, 0].astype('int64')) - 1] = 1
Y_train2 = np.zeros((train_size, 4))
Y_train2[np.arange(train_size), trainlabel[:, 1].astype('int64') - 1] = 1
# matrix1 = np.ones((train_size,25))
Y_train3 = trainlabel_


test_size = testLabel.shape[0]
Y_test1 = np.zeros((test_size, 5))
Y_test1[np.arange(test_size), testLabel[:, 0].astype('int64') - 1] = 1
Y_test2 = np.zeros((test_size, 4))
Y_test2[np.arange(test_size), testLabel[:, 1].astype('int64') - 1] = 1
Y_test3 = testLabel_

# trainData = np.expand_dims(trainData, axis=-1)
# testData = np.expand_dims(testData, axis=-1)
trainData = trainData.reshape((train_size, timestep, -1))
# remain_size = int(((testData.shape[0]*testData.shape[1])/(test_size*timestep)))
# testData = testData.reshape((1,-1))
testData = testData.reshape((test_size, timestep, -1))
# testData = testData.reshape((1,-1))
# valData = valData.reshape((val_size, timestep,-1))
n_input = trainData.shape[2]

def rough_model():
    model_input = Input(shape=(timestep, n_input))
    # mask_input = Input(shape=(mask_d,))
    x = Conv1D(6, 1, activation='relu')(model_input)
    x = Conv1D(6, 1, activation='relu')(x)
    # x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(10, 2, activation='relu')(x)
    x = Conv1D(10, 2, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(16, 3, activation='relu')(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)
    x = Flatten()(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)

    x = Dense(128)(x)
    x = Activation('relu')(x)

    output1 = Dense(5, activation='softmax', name='Bandwidth')(x)

    output2 = Dense(4, activation='softmax', name='Duration')(x)

    # x3 = multiply([x,mask_input])
    # print(mask_input.shape)
    output3 = Dense(num_class, activation='sigmoid', name='Class')(x)

    model = Model(inputs=[model_input], outputs=[output1, output2, output3])
    opt = Adam(clipnorm=1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy'],loss_weights=[1, 1, lambda_value], optimizer=opt)
    return model

start = time.time()

model = rough_model()

# model.fit([trainData,trainmask], [Y_train1, Y_train2, Y_train3],
#           # validation_data = ([np.array(valData), np.array(valmask)], [np.array(Y_val1), np.array(Y_val2), np.array(Y_val3)]),
#           batch_size = 256, epochs = 3, verbose = True, shuffle = True)
hist = model.fit([trainData], [Y_train1, Y_train2, Y_train3],
                 # validation_data = ([np.array(valData), np.array(valmask)], [np.array(Y_val1), np.array(Y_val2), np.array(Y_val3)]),
                 batch_size=512, epochs= 20 ,verbose=True, shuffle=True)

# metrics=[precision,recall,f1,bi_acc]
[_, _, pred] = model.predict(testData, batch_size=256)
Y_test3 = Y_test3.astype(np.float32)
pred = pred.astype(np.float32)
biacc = bi_acc(Y_test3, pred)
pre = precision(Y_test3, pred)
rec = recall(Y_test3, pred)
f1score = f1(Y_test3, pred)

recall_combine_ = recall_combine(Y_test3, pred,25)
f1_combine_ = f1_combine(Y_test3, pred,25)
acc_combine_ = acc_combine(Y_test3, pred,25)
precision_combine_ = precision_combine(Y_test3, pred,25)

rec_each_ = recall_each(Y_test3, pred)
pre_each_ = precision_each(Y_test3, pred)
acc_each_ = acc_each(Y_test3, pred)
f1_each_ = f1_each(Y_test3, pred)

with open('test_log_new1.txt', 'a') as f:
    f.write(str(hist.history) + '\n')
    f.write('------------------------------------------------------')
    f.write('bianary_acc:' + str(biacc) + '\n')
    f.write('precision:' + str(pre) + '\n')
    f.write('recall:' + str(rec) + '\n')
    f.write('f1_score:' + str(f1score) + '\n')
    f.write('---------------combine----------------------------' + '\n')
    f.write('rec:' + str(recall_combine_) + '\n')
    f.write('pre:' + str(precision_combine_) + '\n')
    f.write('acc:' + str(acc_combine_) + '\n')
    f.write('f1:' + str(f1_combine_) + '\n')
    f.write('---------------each--------------------------------' + '\n')
    f.write('rec:' + str(rec_each_) + '\n')
    f.write('pre:' + str(pre_each_) + '\n')
    f.write('acc:' + str(acc_each_) + '\n')
    f.write('f1:' + str(f1_each_) + '\n')
with open('log1.txt', 'w') as f:
    f.write(str(hist.history) + '\n')
end = time.time()
print('{0}s'.format((end-start)))
# result = model.evaluate([testData], [Y_test1, Y_test2, Y_test3])
# print(result)
