import numpy as np
import pandas as pd
import re
from scipy.stats import spearmanr
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import sys
import xgboost as xgb
import random
import cPickle as cp
import time
import datetime

mul_pat = re.compile('(.*)_mult_(.*)')
sub_pat = re.compile('(.*)_sub_(.*)')
add_pat = re.compile('(.*)_add_(.*)')
norm_pat = re.compile('norm_(.*)')

def get_dataset_from_feat_list(feat_list, raw_file):
    t = pd.read_csv(raw_file)
    ids = t.id.values
    has_labels = 'target' in t.columns
    if has_labels:
        labels = t.target.values
        t = t.drop(['id', 'target'], axis=1)
    else:
        t = t.drop(['id'], axis=1)
    mean = np.mean(t, axis = 1)
    std = np.std(t, axis = 1)
    
    t['mean'] = mean
    t['std'] = std

    for feat in feat_list:
        c = get_feat_from_name(t, feat)
        if c is not None:
            t[feat] = c
    if has_labels:
        t['target'] = labels
    t['id'] = ids
    return t


def get_important_feats(fmap, feat_list, rate = None, num_limit=None):
    sorted_items = sorted(fmap.items(), key = lambda x: x[1], reverse = 1)
    print 'asdfasdf'
    if num_limit!=None:
        res = sorted_items[:num_limit]
    elif rate !=None:
        total = 0
        for feat in sorted_items:
            total += feat[1]
        res = [(feat[0], float(feat[1])/total) for feat in sorted_items]
        i = 0
        for j in xrange(len(res)):
            if res[j][1] < rate:
                break
            i+=1
        res  = res[:i]
        print '---- Feature importances beyond %f num=%d -----'% (rate, len(res))
        print res
    # output features 
    selected_feats = []
    for feat in res:
        idx=int(feat[0][1:])
        #print idx
        selected_feats.append(feat_list[idx])
    return selected_feats

def get_prefix_from_param(param, train_name, ntree):
    col = param['bst:colsample_bytree']
    eta = param['bst:colsample_bytree']
    depth = param['bst:max_depth']
    child = param['bst:min_child_weight']
    sub = param['bst:subsample']
    return "xgb_%s_depth%d_col%.3f_sub%.3f_eta%.3f_child%d_%d" % (train_name, depth, col, sub, eta, child, ntree)

def get_train_preds_from_param(param, train_name, ntree):
    return get_prefix_from_param(param, train_name, ntree)+'_train.csv'

def get_test_preds_from_param(param, train_name, ntree):
    return get_prefix_from_param(param, train_name, ntree)+'_test.csv'

def get_model_name_from_param(param, train_name, ntree):
    return get_prefix_from_param(param, train_name, ntree)+'_model.pkl'

def parse_model_file(filename):
    model, ntree, train_name = cp.load(open(filename, 'rb'))
    return model, ntree, train_name

def get_confusion_matrix(truth, pred_proba, norm_class = False):
    ncls = np.max(truth)+1
    if ncls != pred_proba.shape[1]:
        raise Exception('num classes %d %d not match!'%(ncls, pred_proba.shape[1]))
    res = np.zeros((ncls, ncls))
    num_cases = np.zeros(ncls)
    for i in xrange(len(truth)):
        res[truth[i]] += pred_proba[i]
        num_cases[truth[i]] += 1
    if norm_class:
        for i in xrange(ncls):
            res[i] /= num_cases[i]
    return res


def rule_out_dup_feats(train, feat_list, to_skip=None, rho_limit = 0.6):
    new_col_dict = {}
    minMaxScalers = {}
    ss = set(['id', 'target', 'mean', 'std'])
    print 'min max scaling.....'
    for f in train.columns:
        if f not in ss:
            #print np.nonzero(train[f])
            new_col_dict[f] = train[f].astype(float)
            mms = preprocessing.MinMaxScaler()
            new_col_dict[f] = mms.fit_transform(new_col_dict[f])
            #print np.nonzero(train[f] )
            minMaxScalers[f] = mms
    print 'min max scaling done.'
    new_col_dict['mean'] = train['mean']
    new_col_dict['std'] = train['std']
    feat_list = sorted(list(set(feat_list)-set(train.columns)))
    added = set()
    for feat in feat_list:
        #print feat
        c = get_feat_from_name(new_col_dict, feat)
        if c is None: continue
        if to_skip is not None and feat in to_skip:
            new_col_dict[feat] = c
            continue
        has_related = False
        for col, d in new_col_dict.items():
        #for col in added:
        #    d = new_col_dict[col]
            rel = spearmanr(c, d)[0]
            #print col, rel
            if abs(rel) >= rho_limit:
                has_related = True
                print '%s spearman correlation %f breaking out! ' % (feat, rel)
                break
        if not has_related:
            print 'new col %s added' % feat
            new_col_dict[feat] = c
            added.add(feat)
    return new_col_dict.keys()

def get_cross_feats(feat_list = sorted([85, 66, 14, 24, 39, 25, 13, 59, 10, 33])):
    col_list = ['feat_%d'%i for i in feat_list]
    sz = len(col_list)
    res = []
    for i in xrange(sz):
        for j in xrange(i+1, sz):
            res.append('%s_mult_%s' %(col_list[i], col_list[j]))
            res.append('%s_sub_%s' %(col_list[i], col_list[j]))
            res.append('%s_add_%s' %(col_list[i], col_list[j]))
    return res

def gen_stand_feats(t):
    ids = t.id.values
    has_labels = 'target' in t.columns
    if has_labels:
        labels = t.target.values
        t = t.drop(['id', 'target'], axis=1)
    else:
        t = t.drop(['id'], axis=1)
    mean = np.mean(t, axis = 1)
    std = np.std(t, axis = 1)
    
    t['mean'] = mean
    t['std'] = std

    for col in t.columns:
        if col not in ['mean', 'std']:
            new_col = 'norm_%s' % col
            t[new_col] = (t[col]-t['mean']) / t['std']
    t['id'] = ids
    if has_labels:
        t['target'] = labels


def get_stand_feats(feat_list = ['feat_%d'%i for i in xrange(1, 94)]):
    #ss = set(['id','target', 'mean','std'])
    return ['norm_%s' % i for i in feat_list]


def get_feat_from_name(t, feat_name):
    #print feat_name
    ma1 = mul_pat.match(feat_name)
    ma2 = sub_pat.match(feat_name)
    ma3 = add_pat.match(feat_name)
    ma4 = norm_pat.match(feat_name)
    #print feat_name
    res = None
    if ma1 != None:
        ma = ma1
        a = ma.group(1)
        b = ma.group(2)
        res = t[a]*t[b]
    elif ma2 != None:
        ma = ma2
        a = ma.group(1)
        b = ma.group(2)
        res = t[a]-t[b]
    elif ma3 != None:
        ma = ma3
        a = ma.group(1)
        b = ma.group(2)
        res = t[a]+t[b]
    elif ma4 != None:
        ma = ma4
        a = ma.group(1)
        res = (t[a] - t['mean']) / t['std']
    #print 'res',res
    return res

def select_xgb_params(train, labels, rand_times=100, depth_range= None):
    param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
    other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
    print np.unique(labels)
    #print train
    train, test, train_labels, test_labels = cross_validation.train_test_split(train, labels, test_size=0.3)
    dtrain = xgb.DMatrix(train, label=train_labels)
    other['num_class'] = len(np.unique(labels))
    dtest = xgb.DMatrix(test, label=test_labels)
    watchlist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 1000
    best_param = {}
    best_model = None
    best_loss = 100000
    best_ntree = 100
    my_results = []
    for iter in xrange(1, rand_times):
        if depth_range is not None:
            param['bst:max_depth'] = random.randint(depth_range[0], depth_range[1])
        else:
            param['bst:max_depth'] = random.randint(8, 14)
        param['bst:min_child_weight'] = random.randint(1, 7)
        param['bst:subsample'] = random.uniform(0.5, 0.9)
        param['bst:colsample_bytree'] = random.uniform(0.4, 0.9)
        param['bst:eta'] = random.uniform(0.04, 0.13)
        
        print ' - Iter', iter,' -params ', param
        full_param = other.copy()
        full_param.update(param)
        bst,loss,ntree = xgb.train(full_param, dtrain, num_round, watchlist)
        print '      loss', loss, '  ntree', ntree

        preds = bst.predict(dtest)
        cm = get_confusion_matrix(test_labels, preds, norm_class=True)
        print 'Test confusion matrix:'
        print cm

        if loss < best_loss:
            best_loss = loss
            best_ntree = ntree
            best_param = param
            best_model = bst
        my_results.append((loss, ntree, param.copy()))
        print '-------- best loss', best_loss,'  ntree', best_ntree, ' with param ', best_param
    my_results = sorted(my_results, key = lambda x: x[0])
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    cp.dump(my_results, open('hierarchy_param_'+str(st)+'.pkl', 'wb'), protocol=1)
    return my_results

def get_dataset_for_train(f):
    pass
def get_dataset_for_test(f):
    pass