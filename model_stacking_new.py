import pandas as pd
import pandas as pd
import numpy as np
import sys
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
import cPickle as cp
import StackingLinearRegression
reload(StackingLinearRegression)
import otto_utils


# import data
train = pd.read_csv('train.csv', index_col=0)
#test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')


# drop ids and get labels
labels = train.target.values
train.drop(['target'], axis=1)


param_list = cp.load(open('param_select_mult_train3.pkl', 'rb'))
param_list = sorted(param_list, key = lambda x: x[0])[:25]

other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
#watchlist  = [(dtrain,'train')]


# param_list = param_list[:11]

# labels
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)
oneHotEnc = preprocessing.OneHotEncoder()
labels = oneHotEnc.fit_transform(labels.reshape((-1,1)))
labels = labels.toarray()


train_name='feat_sel_mult_train3'
i = 0
preds_train = []
for loss, ntree, param in param_list:
    part_train = pd.read_csv(otto_utils.get_train_preds_from_param(param, train_name, ntree), index_col=0)
    preds_train.append(np.asarray(part_train))
    i+=1
    
lr = 0.1 
decay = 0.05
max_epoch = 500
tol = 0.00007
search_param = True
if search_param:
    best_loss = None
    # split dataset 
    c_preds_train = []
    c_preds_test = []
    num = len(preds_train[0])
    idx = np.random.permutation(num)
    train_idx = idx[: int(num *0.7)]
    test_idx = idx[int(num *0.7):]
    for i in xrange(len(preds_train)):
        c_preds_train.append(preds_train[i][train_idx])
        c_preds_test.append(preds_train[i][test_idx])
        c_train_labels=labels[train_idx]
        c_test_labels=labels[test_idx]
        
    for j in xrange(10):
        lr = np.random.uniform(0.05, 0.11)
        decay = np.random.uniform(0.05, 0.1)
        #tol = np.random.uniform(0.0001, 0.005)
        print '- Search iter%d lr=%.3f,decay=%3.f,tol=%.4f'%( j, lr, decay, tol)
        clf = StackingLinearRegression.StackingLinearRegression(lr, decay, max_epoch, tol)
        clf.fit(c_preds_train, c_train_labels)
        preds= clf.predict(c_preds_test)
        loss = metrics.log_loss( c_test_labels, preds)
        if best_loss == None or best_loss > loss:
            best_loss =loss
            best_lr = lr
            best_decay = decay
            best_tol = tol
        print '- Search iter%d loss%.6f best_loss%.6f lr=%.3f,decay=%3.f,tol=%.4f' %( j, loss, best_loss, lr, decay, tol)
        if j%10==0:
            print '----BEST RESULT best_loss%.6f lr=%.3f,decay=%.3f,tol=%.4f ------ '%(best_loss, best_lr, best_decay, best_tol)
    lr = best_lr
    decay = best_decay
    tol = best_tol
    print '----BEST RESULT best_loss%.6f lr=%.3f,decay=%f,tol=%f ------ '%(best_loss, best_lr, best_decay, best_tol)

clf = StackingLinearRegression.StackingLinearRegression(lr, decay, max_epoch, tol)
    
# train with searched or defined parameters
clf.fit(preds_train, labels)
df = pd.DataFrame(index=sample.id.values)
i = 0
preds_test = []
for loss, ntree, param in param_list:
    part_test = pd.read_csv(otto_utils.get_test_preds_from_param(param, train_name, ntree), index_col=0)
    preds_test.append(np.asarray(part_test))
    i+=1
# predict and output results
preds = clf.predict(preds_test)
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('xgb_'+train_name+'_my_stacking_benchmark.csv', index_label='id')

cp.dump(clf, open('xgb_'+train_name+'_my_stacking.model', 'wb'), protocol = 1)