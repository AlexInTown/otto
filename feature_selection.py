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

# import data
train = pd.read_csv('train.csv')
train_ids = train.id.values
labels = train.target.values
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)
# drop ids and get labels
train = train.drop(['id', 'target'], axis = 1)

feat_list = cp.load(open('mult_feature_list.pkl', 'rb'))
'''
for col in train.columns:
    feat_list.append('%s_mult_mean' % col)
    feat_list.append('%s_mult_std' % col)
'''
mean = np.mean(train, axis = 1)
std = np.std(train, axis = 1)
train['mean'] = mean
train['std'] = std
feat_list.append('mean')
feat_list.append('std')


skip = set(feat_list)
cross_feats = otto_utils.get_cross_feats()
print cross_feats
feat_list.extend(cross_feats)
stand_feats = otto_utils.get_stand_feats()
feat_list.extend(stand_feats)

print ' Before ruling out, sz = %d' % len(feat_list)

feat_list = otto_utils.rule_out_dup_feats(train, feat_list, to_skip=skip, rho_limit=0.5)
print ' After ruling out, sz = %d' % len(feat_list)

for feat in feat_list:
    c = otto_utils.get_feat_from_name(train, feat)
    if c is not None:
        #print c.shape
        train[feat] = c

#train.to_csv('train_mult_raw.csv')
train, test, train_labels, test_labels = cross_validation.train_test_split(train, labels, test_size=0.5)
dtrain = xgb.DMatrix(train, label=train_labels)
dtest = xgb.DMatrix(test, label=test_labels)


param = {'bst:max_depth': 8, 'bst:colsample_bytree': 0.41870944721060444, 'bst:subsample': 0.49849001702707574, 'bst:min_child_weight': 6, 'bst:eta': 0.13}
other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 700
full_param = other.copy()
full_param.update(param)    
bst,loss,ntree = xgb.train(full_param, dtrain, num_round, watchlist)
fmap = bst.get_fscore()
raw_selected_feat_list = otto_utils.get_important_feats(fmap, feat_list, num_limit=500)
cp.dump(raw_selected_feat_list, open('raw_selected_feat_list.pkl', 'wb'), protocol = 1)
set1 = set(otto_utils.get_important_feats(fmap, feat_list, num_limit=100))

watchlist  = [(dtrain,'eval'), (dtest,'train')]
bst,loss,ntree = xgb.train(full_param, dtest, num_round, watchlist)
fmap = bst.get_fscore()
set2 = set(otto_utils.get_important_feats(fmap, feat_list, num_limit=100))

res = set1 & set2
#print feature ids
#to_drop = list(res.difference(set(['norm_feat_%d'%i for i in xrange(1, 94)])))
print sorted(list(res))

selected_feats = res

output_train = otto_utils.get_dataset_from_feat_list(selected_feats, 'train.csv')
#output_train.drop(to_drop, axis = 1)
output_train.to_csv('feat_sel_mult_train6.csv', index =False)
output_train=None

output_test = otto_utils.get_dataset_from_feat_list(selected_feats, 'test.csv')
#output_test.drop(to_drop, axis = 1)
output_test.to_csv('feat_sel_mult_test6.csv', index = False)