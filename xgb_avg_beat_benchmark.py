import pandas as pd
import pandas as pd
import numpy as np
import sys
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
import cPickle as cp
import otto_utils

# training file name
train_name = 'feat_sel_mult_train3'
test_name = 'feat_sel_mult_test3'
# import data
train = pd.read_csv(train_name+'.csv')
test = pd.read_csv(test_name+'.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train_ids = train.id.values


#print labels
train = train.drop(['id','target'], axis=1)
test = test.drop(['id'], axis=1)

'''
mean = np.mean(train, axis = 1)
std = np.std(train, axis = 1)
train['mean'] = mean
train['std'] = std

mean = np.mean(test, axis = 1)
std = np.std(test, axis = 1)
test['mean'] = mean
test['std'] = std
'''

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


dtrain = xgb.DMatrix(train, label=labels)
dtest = xgb.DMatrix(test, label=None)

param_list = cp.load(open('param_select_mult_train3.pkl', 'rb'))
param_list = sorted(param_list, key = lambda x: x[0])[:25]
print param_list

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
watchlist = [(dtrain,'train')]
avg_preds = None

for loss, num_round, param in param_list:
    full_param = other.copy()
    full_param.update(param)
    plst = full_param.items()
    # train on the trainning set
    bst,loss,ntree = xgb.train(full_param, dtrain, num_round, watchlist)

    # dump bst model
    bst_dump_model = otto_utils.get_model_name_from_param(param, train_name, ntree)
    bst.save_model(bst_dump_model)

    # output train predictions
    preds = bst.predict(dtrain)
    preds = pd.DataFrame(preds, index=train_ids, columns=sample.columns[1:])
    preds.to_csv(otto_utils.get_train_preds_from_param(param, train_name, ntree), index_label='id')

    # output test predictions
    preds = bst.predict(dtest)
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(otto_utils.get_test_preds_from_param(param, train_name, ntree), index_label='id')

    if avg_preds is None:
        avg_preds = preds
    else:
        avg_preds += preds
    
avg_preds /= len(param_list)

# create submission file
avg_preds = pd.DataFrame(avg_preds, index=sample.id.values, columns=sample.columns[1:])
avg_preds.to_csv('xgb_avg_benchmark_'+train_name+'.csv', index_label='id')
