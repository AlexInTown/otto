import pandas as pd
import pandas as pd
import numpy as np
import sys
sys.path.append("D:/DEVELOPER_TOOLS/xgboost-master/wrapper");
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
import cPickle as cp

def get_train_from_param(param, ntree):
    col = param['bst:colsample_bytree']
    eta = param['bst:colsample_bytree']
    depth = param['bst:max_depth']
    child = param['bst:min_child_weight']
    sub = param['bst:subsample']
    return "xgb_depth%d_col%.3f_sub%.3f_eta%.3f_child%d_%d_benchmark.csv" % (depth, col, sub, eta, child, ntree)

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

mean = np.mean(train, axis = 1)
std = np.std(train, axis = 1)
train['mean'] = mean
train['std'] = std


mean = np.mean(test, axis = 1)
std = np.std(test, axis = 1)
test['mean'] = mean
test['std'] = std

'''
pca = decomposition.PCA(n_components = 40)
pca.fit(train)
train_pca = pca.transform(train)

for i in xrange(40):
    train['pca_%d'%i] = train_pca[:, i]
test_pca = pca.transform(test)
for i in xrange(20):
    test['pca_%d'%i] = test_pca[:, i]
'''

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


dtrain = xgb.DMatrix(train, label=labels)
dtest = xgb.DMatrix(test, label=None)


#param = {'bst:max_depth': 8, 'bst:colsample_bytree': 0.41870944721060444, 'bst:subsample': 0.49849001702707574, 'bst:min_child_weight': 6, 'bst:eta': 0.230335943854506}
'''param= {'bst:colsample_bytree': 0.8415789385110635,
  'bst:eta': 0.3703558062958862,
  'bst:max_depth': 8,
  'bst:min_child_weight': 6,
  'bst:subsample': 0.4095574265434683}
'''
#param =  {'bst:max_depth': 7, 'bst:colsample_bytree': 0.897509125588231, 'bst:subsample': 0.873318950221093, 'bst:min_child_weight': 6, 'bst:eta': 0.4698537537285306}
#param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}

param_list = cp.load(open('param_select_mult_train3.pkl', 'rb'))[:25]

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
watchlist  = [(dtrain,'train')]
avg_preds = None

for loss, num_round, param in param_list:
    full_param = other.copy()
    full_param.update(param)
    plst = full_param.items()
    # train on the trainning set
    bst,loss,ntree = xgb.train(full_param, dtrain, num_round, watchlist)

    # dump dsp

    # output train predictions
    preds = bst.predict(dtrain)
    preds = pd.DataFrame(preds, index=train_ids, columns=sample.columns[1:])
    preds.to_csv("train_"+get_train_from_param(param, ntree), index_label='id')
    # output test predictions
    preds = bst.predict(dtest)
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(get_train_from_param(param, ntree), index_label='id')

    if avg_preds is None:
        avg_preds = preds
    else:
        avg_preds += preds
    
avg_preds /= len(param_list)

# create submission file
avg_preds = pd.DataFrame(avg_preds, index=sample.id.values, columns=sample.columns[1:])
avg_preds.to_csv('xgb_avg_benchmark_train3.csv', index_label='id')
