import pandas as pd
import pandas as pd
import numpy as np
import sys
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


# import data
train = pd.read_csv('train.csv', index_col=0)
#test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')


# drop ids and get labels
labels = train.target.values
train.drop(['target'], axis=1)


param_list = [(0.464892,
  388,
  {'bst:colsample_bytree': 0.8982472321559289,
   'bst:eta': 0.0726499057464045,
   'bst:max_depth': 9,
   'bst:min_child_weight': 4,
   'bst:subsample': 0.8811612611235187}),
(0.465441,
  322,
  {'bst:colsample_bytree': 0.55270875593484,
   'bst:eta': 0.08046217884098084,
   'bst:max_depth': 10,
   'bst:min_child_weight': 4,
   'bst:subsample': 0.6531447130270357}),
(0.46553,
  461,
  {'bst:colsample_bytree': 0.5366302361900328,
   'bst:eta': 0.07729155155370118,
   'bst:max_depth': 8,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.7019065669513784}),
(0.4659,
  219,
  {'bst:colsample_bytree': 0.7406872560586676,
   'bst:eta': 0.10664938835620626,
   'bst:max_depth': 11,
   'bst:min_child_weight': 4,
   'bst:subsample': 0.8856581787528491}),
(0.465996,
  325,
  {'bst:colsample_bytree': 0.8366932523773002,
   'bst:eta': 0.08916206597121619,
   'bst:max_depth': 9,
   'bst:min_child_weight': 5,
   'bst:subsample': 0.6639194912019956}),
(0.466197,
  276,
  {'bst:colsample_bytree': 0.869674740040372,
   'bst:eta': 0.09258288124705212,
   'bst:max_depth': 9,
   'bst:min_child_weight': 3,
   'bst:subsample': 0.7462811964854872}),
(0.46635,
  315,
  {'bst:colsample_bytree': 0.8867668139142193,
   'bst:eta': 0.0761971730047506,
   'bst:max_depth': 10,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.5805746137229462}),
(0.466797,
  280,
  {'bst:colsample_bytree': 0.6754208230955426,
   'bst:eta': 0.07969828099658503,
   'bst:max_depth': 9,
   'bst:min_child_weight': 1,
   'bst:subsample': 0.7986095783246718}),
(0.466834,
  261,
  {'bst:colsample_bytree': 0.6110072927793166,
   'bst:eta': 0.10011159415056074,
   'bst:max_depth': 10,
   'bst:min_child_weight': 5,
   'bst:subsample': 0.8146995904939873}),
(0.467063,
  363,
  {'bst:colsample_bytree': 0.5236917026970802,
   'bst:eta': 0.10130143030287109,
   'bst:max_depth': 8,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.6845293360554099}),
(0.468003,
  264,
  {'bst:colsample_bytree': 0.7618307386246083,
   'bst:eta': 0.09942926761372282,
   'bst:max_depth': 10,
   'bst:min_child_weight': 5,
   'bst:subsample': 0.7317619775427643}),
(0.468267,
  190,
  {'bst:colsample_bytree': 0.5078649215428852,
   'bst:eta': 0.12959164723390212,
   'bst:max_depth': 11,
   'bst:min_child_weight': 5,
   'bst:subsample': 0.7991674239912825}),
(0.468533,
  254,
  {'bst:colsample_bytree': 0.5963881606988908,
   'bst:eta': 0.14444391434212717,
   'bst:max_depth': 8,
   'bst:min_child_weight': 3,
   'bst:subsample': 0.8527227765966772}),
(0.468729,
  374,
  {'bst:colsample_bytree': 0.7221549470309043,
   'bst:eta': 0.07694229903595157,
   'bst:max_depth': 9,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.5152995556846248}),
(0.468761,
  339,
  {'bst:colsample_bytree': 0.59851235334591,
   'bst:eta': 0.10923887385399847,
   'bst:max_depth': 8,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.8246884368163768}),
(0.46901,
  273,
  {'bst:colsample_bytree': 0.5139248082577734,
   'bst:eta': 0.12937916108345598,
   'bst:max_depth': 8,
   'bst:min_child_weight': 3,
   'bst:subsample': 0.8454156283114604}),
(0.469541,
  406,
  {'bst:colsample_bytree': 0.6205041990211738,
   'bst:eta': 0.07287066098014582,
   'bst:max_depth': 8,
   'bst:min_child_weight': 3,
   'bst:subsample': 0.48456118233083845}),
(0.469756,
  228,
  {'bst:colsample_bytree': 0.8769674814407461,
   'bst:eta': 0.08117083209465441,
   'bst:max_depth': 11,
   'bst:min_child_weight': 4,
   'bst:subsample': 0.5839346433789359}),
(0.469775,
  295,
  {'bst:colsample_bytree': 0.4422611666623005,
   'bst:eta': 0.10330808647559925,
   'bst:max_depth': 8,
   'bst:min_child_weight': 1,
   'bst:subsample': 0.7003156848961816}),
(0.469801,
  326,
  {'bst:colsample_bytree': 0.6238092534945002,
   'bst:eta': 0.07639609838586511,
   'bst:max_depth': 9,
   'bst:min_child_weight': 2,
   'bst:subsample': 0.5727941903011948}),
(0.470169,
  182,
  {'bst:colsample_bytree': 0.723570738914302,
   'bst:eta': 0.1138948372297852,
   'bst:max_depth': 11,
   'bst:min_child_weight': 6,
   'bst:subsample': 0.6800040176803429}),
(0.470237,
  233,
  {'bst:colsample_bytree': 0.6706689191818904,
   'bst:eta': 0.09621163582569642,
   'bst:max_depth': 10,
   'bst:min_child_weight': 2,
   'bst:subsample': 0.6307041906020502})]

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
watchlist  = [(dtrain,'train')]


# param_list = param_list[:11]

# labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

col_list = ['Class_%d'%i for i in xrange(1, 10)]
df = pd.DataFrame(index=train_ids)
i = 0
preds_train = []
for loss, ntree, param in param_list:
    part_train = pd.read_csv("train_"+get_train_from_param(param, ntree), index_col=0)
    #part_train.drop(['id'], axis=1)
    #for col in col_list:
    #    df[col+'_%d'%i] = part_train[col]
    preds_train.append(np.asarry(part_train))
    i+=1
    
cv_search_param = False
if cv_search_param:
    for j in xrange(100):
        pass
else:
    clf = linear_model.LogisticRegression('l1', C=0.1)
train = np.asarray(df)
clf.fit(train, labels)

df = pd.DataFrame(index=sample.id.values)
i = 0
preds_test = []
for loss, ntree, param in param_list:
    part_test = pd.read_csv(get_train_from_param(param, ntree), index_col=0)
    #for col in col_list:
    #    df[col+'_%d'%i] = part_test[col]
    preds_test.append(np.asarray(part_test))
    i+=1
test = np.asarray(df)
df = None
preds = clf.predict_proba(test)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('xgb_22_stacking_l1_C0.1_benchmark.csv', index_label='id')

cp.dump(clf, open('xgb_22_stacking_l1_C0.1_lr.model', 'wb'), protocol = 1)