import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import binary_accuracy
from sklearn.metrics import recall_score, precision_score, f1_score
from tensorflow.keras.metrics import Recall, Precision, FalsePositives
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
# tf.config.run_functions_eagerly(run_eagerly=True)

def precision(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = Precision()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def recall(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = Recall()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


def f1(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def bi_acc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return binary_accuracy(y_true, y_pred).numpy().mean()


def fpr(y_true, y_pred):
    m = FalsePositives()
    m.update_state(y_true, y_pred)
    return m.result().numpy() / y_true.shape[0]


def acc_combine(y_true, y_pred, label_size):
    rtn = {}
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for n in range(1, label_size + 1):
        cond = (y_true.sum(axis=1) == n)
        cond_n = cond.sum()
        if cond_n == 0:
            acc = 0
        else:
            acc = bi_acc(y_true[cond], y_pred[cond])
        rate = cond_n / y_true.shape[0]
        rtn[n] = (acc, rate)
    return rtn


def precision_combine(y_true, y_pred, label_size):
    rtn = {}
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for n in range(1, label_size + 1):
        cond = (y_true.sum(axis=1) == n)
        cond_n = cond.sum()
        if cond_n == 0:
            acc = 0
        else:
            acc = precision(y_true[cond], y_pred[cond])
        rate = cond_n / y_true.shape[0]
        rtn[n] = (acc, rate)
    return rtn


def recall_combine(y_true, y_pred, label_size):
    rtn = {}
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    for n in range(1, label_size + 1):
        cond = (y_true.sum(axis=1) == n)
        cond_n = cond.sum()
        if cond_n == 0:
            acc = 0
        else:
            acc = recall(y_true[cond], y_pred[cond])
        rate = cond_n / y_true.shape[0]
        rtn[n] = (acc, rate)
    return rtn


def f1_combine(y_true, y_pred, label_size):
    rtn = {}
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    for n in range(1, label_size + 1):
        cond = (y_true.sum(axis=1) == n)
        cond_n = cond.sum()
        if cond_n == 0:
            acc = 0
        else:
            acc = f1(y_true[cond], y_pred[cond])
        rate = cond_n / y_true.shape[0]
        rtn[n] = (acc, rate)
    return rtn


def fnr_category(model: Model, X, y, category_idxs: list):
    fnr_list = []
    for idxs in category_idxs:
        fn = 0
        p = 0
        for i in idxs:
            cond = (y[:, i] == 1)
            prob = model.predict(X[cond])[:, i]
            p += prob.shape[0]
            fn += prob[prob < 0.5].shape[0]
        fnr = fn / p
        fnr_list.append(fnr)
    return fnr_list


def acc_each(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    metric_list = []
    for i in range(y_true.shape[1]):
        #         cond = (y_true[:, i] == 1)
        #         m = bi_acc(y_true[cond], y_prob[cond])
        m = bi_acc(y_true[:, i].reshape(-1, 1), y_prob[:, i].reshape(-1, 1))
        metric_list.append(m)
    return metric_list


def precision_each(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    metric_list = []
    for i in range(y_true.shape[1]):
        #         cond = (y_true[:, i] == 1)
        #         m = precision(y_true[cond], y_prob[cond])
        m = precision(y_true[:, i].reshape(-1, 1), y_prob[:, i].reshape(-1, 1))
        metric_list.append(m)
    return metric_list


def recall_each(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    metric_list = []
    for i in range(y_true.shape[1]):
        #         cond = (y_true[:, i] == 1)
        #         m = recall(y_true[cond], y_prob[cond])
        m = recall(y_true[:, i].reshape(-1, 1), y_prob[:, i].reshape(-1, 1))
        metric_list.append(m)
    return metric_list


def f1_each(y_true, y_prob):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    metric_list = []
    for i in range(y_true.shape[1]):
        #         cond = (y_true[:, i] == 1)
        #         m = f1(y_true[cond], y_prob[cond])
        m = f1(y_true[:, i].reshape(-1, 1), y_prob[:, i].reshape(-1, 1))
        metric_list.append(m)
    return metric_list


def fpr_each(y_true, y_prob):
    metric_list = []
    for i in range(y_true.shape[1]):
        m = fpr(y_true[:, i].reshape(-1, 1), y_prob[:, i].reshape(-1, 1))
        metric_list.append(m)
    return metric_list


def euclidean(a, b):
    return np.linalg.norm(a - b)


def plot_confusion_matrix(cm, classes,
                          path,
                          title=None,
                          cmap=plt.cm.YlGnBu):
    #     plt.rc('font', family='Times New Roman')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    cb = plt.colorbar()
    cb.set_label(title, fontsize=18, fontproperties='Times New Roman', )
    tick_marks = np.arange(0, len(classes), 2)
    plt.xticks(tick_marks, tick_marks, rotation=0, fontproperties='Times New Roman', )
    plt.yticks(tick_marks, tick_marks, fontproperties='Times New Roman', )

    #     thresh = np.min(cm, axis=1)
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, round(cm[i, j], 2),
    #                  horizontalalignment="center",
    #                  color="black" if cm[i, j] > thresh[i] else "red")

    plt.ylabel('Device number', fontproperties='Times New Roman', fontsize=14)
    plt.xlabel('Device number', fontproperties='Times New Roman', fontsize=14)
    plt.tight_layout()
    plt.savefig(path, format='pdf')
    plt.show()

