import pandas as pd
import random
import numpy as np
import sys
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
import cPickle as cp

def get_best_log_loss( model, dtest, test_labels, n_tree_limit):
    # predict on test set
    oneHotEnc = preprocessing.OneHotEncoder()
    test_labels = np.reshape(test_labels, (test_labels.size, 1))
    oneHotEnc.fit(test_labels)
    truth=oneHotEnc.fit_transform(test_labels)
    best_ntree = 1
    best_loss = 1000000
    for ntree in xrange(n_tree_limit, n_tree_limit-10, -1):
        preds = bst.predict(dtest, ntree_limit=ntree)
        loss = metrics.log_loss(truth, preds)
        if loss < best_loss:
            best_loss = loss
            best_ntree= ntree
    return best_loss, best_ntree


# import data
train = pd.read_csv('feat_sel_mult_train6.csv')

# drop ids and get labels
labels = train.target.values
#print labels
train = train.drop(['id','target'], axis=1)
mean = np.mean(train, axis = 1)
std = np.std(train, axis = 1)
#max_feat = np.argmax(train, axis = 1)
#min_feat = np.argmax(train, axis = 1)

#train['mean'] = mean
#train['std'] = std
train['max_feat'] = mean
train['min_feat'] = std
'''
pca = decomposition.PCA(n_components = 40)
pca.fit(train)
train_pca = pca.transform(train)

for i in xrange(40):
    train['pca_%d'%i] = train_pca[:, i]
'''
'''
col_list = ['feat_%d'%i for i in [85, 66, 14]]
for i in xrange(len(col_list)):
    a = train[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = train[col_list[j]]
        train['%s_%s' % (col_list[i], col_list[j])] = a-b
'''

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# split trainset and testset
train, test, train_labels, test_labels = cross_validation.train_test_split(train, labels, test_size=0.3)
dtrain = xgb.DMatrix(train, label=train_labels)
dtest = xgb.DMatrix(test, label=test_labels)

# for cross validation
#dtrain = xgb.DMatrix(train, label=labels)

# train a xgboost tree classifier
param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
#param = {'bst:max_depth': 9, 'bst:colsample_bytree': 0.9369305522074396, 'bst:subsample': 0.45423912239653175, 'bst:min_child_weight': 2, 'bst:eta': 0.3827701004451156}
#param = {'bst:max_depth': 8, 'bst:colsample_bytree': 0.41870944721060444, 'bst:subsample': 0.49849001702707574, 'bst:min_child_weight': 6, 'bst:eta': 0.230335943854506}

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 700

best_param = {}
best_model = None
best_loss = 100000
best_ntree = 100
my_results = []
for iter in xrange(1, 1000):
    '''
    param['bst:max_depth'] = random.randint(8, 11)
    param['bst:min_child_weight'] = random.randint(1, 6)
    param['bst:subsample'] = random.uniform(0.4, 0.9)
    param['bst:colsample_bytree'] = random.uniform(0.4, 0.9)
    param['bst:eta'] = random.uniform(0.07, 0.18)
    '''
    param['bst:max_depth'] = random.randint(9, 14)
    param['bst:min_child_weight'] = random.randint(1, 6)
    param['bst:subsample'] = random.uniform(0.6, 0.9)
    param['bst:colsample_bytree'] = random.uniform(0.4, 0.9)
    param['bst:eta'] = random.uniform(0.04, 0.13)
    
    '''
    param['bst:max_depth'] = int(np.random.normal(9.5, 1.7))
    param['bst:min_child_weight'] = random.randint(1, 6)
    param['bst:subsample'] = random.uniform(0.4, 0.88)
    param['bst:colsample_bytree'] = random.uniform(0.4, 0.99)
    param['bst:eta'] = np.random.normal(0.10, 0.15)
    '''
    print ' - Iter', iter,' -params ', param
    full_param = other.copy()
    full_param.update(param)
    
    '''
    #plst = full_param.items()
    results = xgb.cv(full_param, dtrain, num_round, nfold=5, show_stdv=1, seed=0)
    for line in results:
        print line
    #print results
    '''
    
    bst,loss,ntree = xgb.train(full_param, dtrain, num_round, watchlist)
    preds = bst.predict(dtrain)
    cm = otto_utils.get_confusion_matrix(train_labels, preds, norm_class=True)
    print 'Train confusion matrix:'
    print cm

    preds = bst.predict(dtest)
    cm = otto_utils.get_confusion_matrix(test_labels, preds, norm_class=True)
    print 'Test confusion matrix:'
    print cm
    break

    #loss, ntree = get_best_log_loss(bst, dtest, test_labels, num_round)
    print '      loss', loss, '  ntree', ntree
    if loss < best_loss:
        best_loss = loss
        best_ntree = ntree
        best_param = param
        best_model = bst
    my_results.append((loss, ntree, param.copy()))
    # print 
    #if iter % 10 == 0:
    print '-------- best loss', best_loss,'  ntree', best_ntree, ' with param ', best_param

sort_results = sorted(my_results, key=lambda x: x[0])

cp.dump(sorted_results, open('feat_mult_sel_params.pkl', 'wb'), protocol = 1)