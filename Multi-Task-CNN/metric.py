# -*- coding: utf-8 -*-  
'''
Created on 2020年7月10日
自定义评价标准
@author: lw 刘伟
'''
from keras import backend as K

def jacc_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((intersection + 1e-6) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1e-6))

#精确率评价指标
def precision(y_true,y_pred):    
    TP=K.sum(y_true*K.round(y_pred))
    TN=K.sum((1-y_true)*(1-K.round(y_pred)))
    FP=K.sum((1-y_true)*K.round(y_pred))
    FN=K.sum(y_true*(1-K.round(y_pred)))
    precision=TP/(TP+FP+ 1e-6)
    return precision

#召回率评价指标
def recall(y_true,y_pred):  
    TP=K.sum(y_true*K.round(y_pred))
    TN=K.sum((1-y_true)*(1-K.round(y_pred)))
    FP=K.sum((1-y_true)*K.round(y_pred))
    FN=K.sum(y_true*(1-K.round(y_pred)))
    recall=TP/(TP+FN+ 1e-6)
    return recall

#F1-score评价指标
def f1score(y_true,y_pred):    
    TP=K.sum(y_true*K.round(y_pred))
    TN=K.sum((1-y_true)*(1-K.round(y_pred)))
    FP=K.sum((1-y_true)*K.round(y_pred))
    FN=K.sum(y_true*(1-K.round(y_pred)))
    precision=TP/(TP+FP+ 1e-6)
    recall=TP/(TP+FN+ 1e-6)
    F1score=2*precision*recall/(precision+recall+ 1e-6)
    return F1score
 
def accuracy(y_true, y_pred):
    TP=K.sum(y_true*K.round(y_pred))
    TN=K.sum((1-y_true)*(1-K.round(y_pred)))
    FP=K.sum((1-y_true)*K.round(y_pred))
    FN=K.sum(y_true*(1-K.round(y_pred)))
    accuracy = (TP+TN)/(TP + FP + FN + TN + 1e-6)
    return accuracy

def miou(y_true, y_pred):
    TP=K.sum(y_true*K.round(y_pred))
    TN=K.sum((1-y_true)*(1-K.round(y_pred)))
    FP=K.sum((1-y_true)*K.round(y_pred))
    FN=K.sum(y_true*(1-K.round(y_pred)))
    iou0 = TN/(FP + TN + FN + 1e-6)
    iou1 = TP/(FP + TP + FN+ 1e-6)
    mean_iou = (iou1 + iou0)/2

    return mean_iou

def binary_accuracy(y_true, y_pred):
    #accuracy = m.binary_accuracy(y_true, y_pred)
    accuracy =  K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    return accuracy
