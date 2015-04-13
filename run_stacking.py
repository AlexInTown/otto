# import stacking
# reload(stacking)
# from stacking import Stacking


import numpy as np
import cPickle as cp
import sys
import xgboost as xgb
import pandas as pd

from sklearn import cross_validation, ensemble, linear_model, preprocessing
from sklearn.ensemble import stacking
import xgb_wrapper
reload(xgb_wrapper)
from xgb_wrapper import XgbWrapper

# import data
train = pd.read_csv('train.csv')
train_ids = train.id.values
labels = train.target.values
train = train.drop(['id','target'], axis = 1)

test = pd.read_csv('test.csv')
test = test.drop(['id'], axis = 1)

sample = pd.read_csv('sampleSubmission.csv')

mean = train.mean(axis = 1)
std = train.std(axis = 1)
train['mean'] = mean 
train['std'] = std

mean = test.mean(axis = 1)
std = test.std(axis = 1)
test['mean'] = mean
test['std'] = std

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)


train = np.asarray(train)
test = np.asarray(test)

# load useful parameters
param_list = cp.load(open('./data/param_raw_train_22.pkl', 'rb'))
param_list = sorted(param_list, key = lambda x: x[0])[:3]
print param_list

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}

# initialize xgboost models
clfs = [XgbWrapper(param, ntree) for loss, ntree, param in param_list]

# adding random forest models

# adding linear models

# adding neural network models

n_folds = 5

# Generate k stratified folds of the training data.
skf = list(cross_validation.StratifiedKFold(labels, n_folds))


# Stacking
stk = stacking.Stacking(linear_model.LogisticRegression, clfs, skf, stackingc=False)
stk.fit(train, labels)

preds = stk.predict_proba(test)

preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('real_stacking_raw_xgb_22.csv', index_label='id')