import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from utils1 import precision,recall,f1,bi_acc,acc_each,precision_each,recall_each,f1_each,acc_combine,precision_combine,recall_combine,f1_combine
# tf.config.run_functions_eagerly(run_eagerly=True)
# from tensorflow.compat.v1 import ConfigProto

# from tensorflow.compat.v1 import InteractiveSession
# # 定义TensorFlow配置
# config = ConfigProto()
# # 配置GPU内存分配方式，按需增长，很关键
# config.gpu_options.allow_growth = True
# # 在创建session的时候把config作为参数传进去
# session = InteractiveSession(config=config)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 1024
EPOCHS = 10 # use early stopping
seq_len = 20
FOLDS = 10



def data_load_and_filter():
    feature = np.load('data2.npy',allow_pickle=True)
    label = np.load('label2.npy',allow_pickle=True)
    # feature = np.array(feature)
    print(np.shape(feature))
    x_len = feature.shape[0]
    length = feature.shape[0]*feature.shape[1]
    feature = feature.reshape((1,-1))
    # print('size:',feature.size)

    # feature = feature[:length - length %(3*seq_len)]
    feature = np.reshape(feature,(x_len,seq_len,4))
    print(np.shape(feature))

    # packet sizes
    X_size = feature[:,:,0]
    # inter-arrival times
    # 对到达间隔进行log对数换算
    X_time = feature[:,:,1]
    # for i in range(0,feature.shape[1]):
    #     print(X_time[7][i])

    # X_time[np.where(X_time!=0.0)] = - np.log(X_time[np.where(X_time!=0.0 )])
    # for i in range(0,feature.shape[1]):
    #     print(X_time[7][i])
    # direction
    X_direc = feature[:,:,2]

    X_payload = feature[:,:,3]
    # print('x3 shape:',X_direc.shape)
    for i in range (0,feature.shape[0]):
        for j in range(0,feature.shape[1]):
            if(X_direc[i][j] ==0.0):
                X_size[i][j] *=(-1)
    y = np.array(label)
    print("Shape of X_time =", np.shape(X_time))
    print("Shape of X_size =", np.shape(X_size))
    # print("Shape of X3 =", np.shape(X3))
    print("Shape of y =", np.shape(y))
    n_samples = np.shape(feature)[0]
    return X_time, X_size, y, X_payload
def DLClassification(X_train, X_test, y_train, y_test,dropout):
    X_train = np.stack([X_train], axis=2)  #CNN输入必须是3维，而原始数据只有两维，此处进行扩维
    X_test = np.stack([X_test], axis=2)

    # if you dont have newest keras version, you might have to remove restore_best_weights = True
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min')
    model = Sequential()
    # seq_len = 25  # we have a time series of 100 payload sizes
    n_features = 1  # 1 feature which is packet size
    model.add(Conv1D(200, 3, activation='relu', input_shape=(seq_len, n_features)))
    model.add(BatchNormalization())
    model.add(Conv1D(400, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GRU(200))
    model.add(Dropout(dropout))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(25, activation='sigmoid'))
    #binary_accuracy
    #,acc_combine,precision_combine,recall_combine,f1_combine
    model.compile(loss='binary_crossentropy',optimizer='adam')
    #metrics=[precision,recall,f1,bi_acc]
    # model.summary()
    hist = model.fit(X_train, y_train,epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=True, shuffle=True)
    # callbacks = [early_stopping]
    # print(hist.history)
    pred = model.predict(X_test,batch_size=BATCH_SIZE)

    y_test = tf.cast(y_test, tf.float32)
    pred = tf.cast(pred, tf.float32)

    biacc = bi_acc(y_test, pred)
    pre = precision(y_test, pred)
    rec = recall(y_test, pred)
    f1score = f1(y_test, pred)



    recall_combine_ = recall_combine(y_test,pred,25)
    f1_combine_ = f1_combine(y_test,pred,25)
    acc_combine_ = acc_combine(y_test,pred,25)
    precision_combine_ = precision_combine(y_test,pred,25)
    #acc_each,precision_each,recall_each,f1_each


    rec_each_ = recall_each(y_test,pred)
    pre_each_ = precision_each(y_test,pred)
    acc_each_ = acc_each(y_test,pred)
    f1_each_ = f1_each(y_test,pred)
    with open('train_log2.txt', 'a') as f:
        f.write(str(hist.history) + '\n')
        f.write('---------------overall-------------' + '\n')
        f.write('rec:' + str(rec) + '\n')
        f.write('pre:' + str(pre) + '\n')
        f.write('acc:' + str(biacc) + '\n')
        f.write('f1:' + str(f1score) + '\n')
        f.write('---------------combine-------------'+'\n')
        f.write('rec:' + str(recall_combine_) + '\n')
        f.write('pre:' + str(precision_combine_) + '\n')
        f.write('acc:' + str(acc_combine_) + '\n')
        f.write('f1:' + str(f1_combine_) + '\n')
        f.write('---------------each-------------'+'\n')
        f.write('rec:' + str(rec_each_) + '\n')
        f.write('pre:' + str(pre_each_) + '\n')
        f.write('acc:' + str(acc_each_) + '\n')
        f.write('f1:' + str(f1_each_) + '\n')
    with open('log2.txt', 'a') as f:
        f.write(str(hist.history) + '\n')

    # return model.predict(X_test)
    return pred
def loadData2():
    data2 = np.load('../data3.npy',allow_pickle=True)
    label2 = np.load('../label3.npy',allow_pickle=True)
    col_time = data2[:,:seq_len]
    for i in range(col_time.shape[0]):
        pivot = col_time[i][0]
        for j in range(1,seq_len):
            if col_time[i][j] - pivot > 0:
                col_time[i][j] = -np.log10(col_time[i][j] - pivot)
        col_time[i][0] = 0

    col_direct = data2[:,30:(30+seq_len)]
    np.where(col_direct == 0,-1,1)
    col_packet_size = data2[:,60:(60+seq_len)]
    # col_payload_size = data2[:,120:120+seq_len]

    data = np.concatenate([col_packet_size,col_time,col_direct],axis=1)

    np.save('data2.npy',data)
    np.save('label2.npy',label2)



if __name__ == "__main__":
    # datasetfile = r'/home/lry/fingerprint/25dev_1mo_raw.csv'
    # loadData2()
    # kf = KFold(n_splits=FOLDS, shuffle=True)

    # try a variety of min conn settings for graph
    # accuracies = []

    X1, X2, y, X3 = data_load_and_filter()
    X_time_train ,X_time_test ,y_time_train,y_time_test = train_test_split(X1,y,test_size= 0.5,random_state = 98)
    X_size_train, X_size_test, y_size_train, y_size_test = train_test_split(X2, y, test_size=0.5, random_state= 98)
    X_payload_train, X_payload_test, y_payload_train, y_payload_test = train_test_split(X3, y, test_size=0.5, random_state= 98)
    pred1 = DLClassification(X_time_train, X_time_test,y_time_train, y_time_test, 0)
    pred2 = DLClassification(X_size_train, X_size_test, y_size_train, y_size_test, 0)
    pred3 = DLClassification(X_payload_train, X_payload_test, y_payload_train, y_payload_test, 0.25)
    predictions123 = (pred1 * 0.25 + pred2 * 0.7 + pred3 * 0.05)
    y_test = y_time_test
    # nn_acc123 = 1. * np.sum([np.argmax(x) for x in predictions123] == y_test) / len(y_test)
    # print("Ensemble CNN-RNN ACCURACY: %s" % (nn_acc123))
    pred = predictions123
    biacc = bi_acc(y_test, pred)
    pre = precision(y_test, pred)
    rec = recall(y_test, pred)
    f1score = f1(y_test, pred)

    recall_combine_ = recall_combine(y_test, pred, 25)
    f1_combine_ = f1_combine(y_test, pred, 25)
    acc_combine_ = acc_combine(y_test, pred, 25)
    precision_combine_ = precision_combine(y_test, pred, 25)
    # acc_each,precision_each,recall_each,f1_each

    rec_each_ = recall_each(y_test, pred)
    pre_each_ = precision_each(y_test, pred)
    acc_each_ = acc_each(y_test, pred)
    f1_each_ = f1_each(y_test, pred)
    with open('train_log2.txt', 'a') as f:
        f.write('--------ensemble:-overall-------------' + '\n')
        f.write('rec:' + str(rec) + '\n')
        f.write('pre:' + str(pre) + '\n')
        f.write('acc:' + str(biacc) + '\n')
        f.write('f1:' + str(f1score) + '\n')
        f.write('---------------combine-------------' + '\n')
        f.write('rec:' + str(recall_combine_) + '\n')
        f.write('pre:' + str(precision_combine_) + '\n')
        f.write('acc:' + str(acc_combine_) + '\n')
        f.write('f1:' + str(f1_combine_) + '\n')
        f.write('---------------each-------------' + '\n')
        f.write('rec:' + str(rec_each_) + '\n')
        f.write('pre:' + str(pre_each_) + '\n')
        f.write('acc:' + str(acc_each_) + '\n')
        f.write('f1:' + str(f1_each_) + '\n')


