import pandas as pd
import random
import numpy as np
import sys
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
import cPickle as cp
import re
import otto_utils
reload(otto_utils)
import hierarchy
reload(hierarchy)
# import data
train = pd.read_csv('train.csv')
train_ids = train.id.values
labels = train.target.values
train = train.drop(['id','target'], axis = 1)


test = pd.read_csv('test.csv')
test = test.drop(['id'], axis = 1)

sample = pd.read_csv('sampleSubmission.csv')

#for col in train.columns:
#    train[col] = train[col].astype('float')
#   test[col] = test[col].astype('float')

mean = train.mean(axis = 1)
std = train.std(axis = 1)
train['mean'] = mean 
train['std'] = std

mean = test.mean(axis = 1)
std = test.std(axis = 1)
test['mean'] = mean
test['std'] = std

cls_sets = [set(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_7']), set(['Class_5', 'Class_6', 'Class_8', 'Class_9'])]
L2C = hierarchy.Level2Classification(train, labels, cls_sets)
L2C.train(select_param = True)
#preds = L2C.predict(test)
#L2C.load_param_select_results()

preds = L2C.avg_predict(test, num_model = 5)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('hierarchy_avg_5_beat_benchmark.csv', index_label='id')

preds = L2C.avg_predict(test, num_model = 10)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('hierarchy_avg_10_beat_benchmark.csv', index_label='id')

preds = L2C.avg_predict(test, num_model = 15)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('hierarchy_avg_15_beat_benchmark.csv', index_label='id')


preds = L2C.avg_predict(test, num_model = 20)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('hierarchy_avg_20_beat_benchmark.csv', index_label='id')
