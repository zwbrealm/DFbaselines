import os
import sys
from model import Fs_net
from dataset import dataset
from sklearn.model_selection import train_test_split
from utils import precision,recall,f1,bi_acc,acc_each,precision_each,recall_each,f1_each,acc_combine,precision_combine,recall_combine,f1_combine
from tensorflow.python.keras.metrics import binary_accuracy
import tensorflow  as tf
# from tf.keras import backend as K
# from dataProcess import loadData
import numpy as np
# tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

n_steps = 25
n_inputs = 1
batch_size = 512
label_size = 25
feature = np.load(r'data2.npy',allow_pickle=True)
label = np.load(r'label2.npy',allow_pickle=True)
X_train, X_test, Y_train, Y_test = train_test_split(feature, label, test_size=0.4, random_state=9)

X1 = tf.compat.v1.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
Y1 = tf.compat.v1.placeholder(tf.int32, [batch_size,label_size])
X2 = tf.compat.v1.placeholder(tf.float32, [X_test.shape[0], n_steps, n_inputs])
Y2 = tf.compat.v1.placeholder(tf.int32, [X_test.shape[0],label_size])
fs_net = Fs_net(X1, Y1)
loss, logits = fs_net.build_fs_net_loss()
lr = 1e-4
optimizer  = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss)

init_op =tf.compat.v1.global_variables_initializer()
n_epoch = 3
batch_num = X_train.shape[0] // batch_size
print('batch_num:----------',batch_num,'----------------------')
# batch_num = 6
training_steps_per_epoch = 50
cnt = 0
saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    sess.run(tf.compat.v1.local_variables_initializer())
    train_dataset_X = dataset()
    train_dataset_Y = dataset()
    for epoch in range(n_epoch):
        sum_acc,sum_pre,sum_rec,sum_f1 = 0,0,0,0
        for batch in range(batch_num):
            X_batch, y_batch = train_dataset_X.next_batch(X_train,batch_size),train_dataset_Y.next_batch(Y_train,batch_size)
            X_batch = np.expand_dims(X_batch,-1)
            y_batch = y_batch.astype(np.int32)
            _,loss_= sess.run([train_op,loss],feed_dict={X1:X_batch, Y1:y_batch})
            cnt += 1
            if cnt % training_steps_per_epoch == 0:
                print("{} step {} loss {} ".format(batch_num,cnt,loss_))
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "./checkpoint_dir/MyModel")
    steps_per_epoch = X_test.shape[0]//batch_size
    # steps_per_epoch = 6
    correct_cnt = 0
    test_dataset_X = dataset()
    test_dataset_Y = dataset()
    num_test_examples = X_test.shape[0]

    for test_steps in range(steps_per_epoch):
        X_batch, y_batch = test_dataset_X.next_batch(X_test, batch_size), test_dataset_Y.next_batch(Y_test, batch_size)
        X_batch = np.expand_dims(X_batch, axis=-1)
        logits_= sess.run(logits,feed_dict={X1: X_batch, Y1: y_batch})
        if(test_steps == 0 ):#and epoch ==n_epoch-1
            pred_logits = logits_
        else:  #and epoch ==n_epoch-1
            pred_logits = np.concatenate((pred_logits,logits_),axis=0)

test_conditions = pred_logits >0.5
test_result_Y = test_conditions.astype(np.int32)
test_true_Y = (Y_test[0:steps_per_epoch*batch_size]).astype(np.int32)

bi_accuracy_score = bi_acc(test_true_Y, test_result_Y)
pre = precision(test_true_Y, test_result_Y)
rec = recall(test_true_Y, test_result_Y)
F1_score = f1(test_true_Y, test_result_Y)

# test_result_Y = tf.cast(test_conditions, tf.int32)
acc_each_ = acc_each(test_true_Y, test_result_Y)
precision_each_ = precision_each(test_true_Y,test_result_Y)
recall_each_ = recall_each(test_true_Y, test_result_Y)
f1_each_ = f1_each(test_true_Y, test_result_Y)

acc_combine_ = acc_combine(test_true_Y, test_result_Y)
precision_combine_ = precision_combine(test_true_Y, test_result_Y)
recall_combine_ = recall_combine(test_true_Y, test_result_Y)
f1_combine_ = f1_combine(test_true_Y, test_result_Y)


with open('test_log2.txt','a') as f:
# f.write('----epoch:---'+str(epoch)+'-----------------step:'+str(test_steps)+'-------------'+'\n')
    f.write(':acc ' + str(bi_accuracy_score) + '\n')
    f.write(':precision ' + str(pre) + '\n')
    f.write(':recall ' + str(rec) + '\n')
    f.write(':f1 ' + str(F1_score) + '\n')

    f.write('---------------------------------each------------------------------------------------'+'\n')
    for i in range(25):
        f.write('acc_each type:'+str(i + 1)+ '  '+str(acc_each_[i]) + '\n')
    for i in range(25):
        f.write('precision_each type:'+str(i + 1)+'  '+ str(precision_each_[i]) + '\n')
    for i in range(25):
        f.write(':recall_each type' + str(i + 1) +'  '+ str(recall_each_[i]) + '\n')
    for i in range(25):
        f.write(':f1_each type' + str(i + 1) +'  '+ str(f1_each_[i]) + '\n')
    f.write('---------------------------------combine------------------------------------------------' + '\n')
    for i in range(25):
        f.write('acc_combine divice nums:'+str(i+1)+ '  acc:'+str(acc_combine_[i+1][0]) +'  rate:'+str(acc_combine_[i+1][1])+ '\n')
    for i in range(25):
        f.write('precision_combine divice nums:'+str(i+1)+ '  acc:'+str(precision_combine_[i+1][0]) +'  rate:'+str(precision_combine_[i+1][1])+ '\n')
    for i in range(25):
        f.write('rec_combine divice nums:'+str(i+1)+ '  acc:'+str(recall_combine_[i+1][0]) +'  rate:'+str(recall_combine_[i+1][1])+ '\n')
    for i in range(25):
        f.write(':f1_combine divice nums' +str(i+1)+ '  acc:'+ str(f1_combine_[i+1][0]) +'  rate:'+ str(recall_combine_[i+1][1])+'\n')
print('---------------------------------------------------------')
