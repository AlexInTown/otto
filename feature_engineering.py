import numpy as np
import pandas as pd
from sklearn import feature_extraction, preprocessing, cluster
from scipy.stats import spearmanr
import UniFind
# import data
train = pd.read_csv('train.csv')


# drop ids and get labels
labels = train.target.values


# encode labels 
#lbl_enc = preprocessing.LabelEncoder()
#labels = lbl_enc.fit_transform(labels)
print 'processing training set....'
#df = pd.DataFrame(index=train['id'])
col_list = list(train.columns.values)[1:-1]
#new_col_list = []
new_col_dict = {}
standardScalers = []
minMaxScalers = []
for col in col_list:
    #ss = preprocessing.StandardScaler(copy=False)
    #new_col_dict[col] = ss.fit_transform(train[col])
    #standardScalers.append(ss)
    
    mms = preprocessing.MinMaxScaler(copy=False)
    new_col_dict[col] = ss.fit_transform(train[col])
    minMaxScalers.append(mms)

for i in xrange(27, len(col_list)):
    a = new_col_dict[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = new_col_dict[col_list[j]]
        new_col = '%s_mult_%s' % (col_list[i], col_list[j])
        #new_col_list.append(new_col) 
        visited = set()
        c = a*b
        has_related = False
        for col, d in new_col_dict.items():
            rel = spearmanr(c, d)[0]
            #print 'spearman correlation %.3f ' % rel
            if abs(rel) >= 0.45:
                has_related = True
                print '%s spearman correlation %.3f breaking out! ' % (new_col, rel)
                break
        if not has_related:
            print 'new col %s added' % new_col
            new_col_dict[new_col] = c
            
for i in xrange(len(col_list)):
    a = new_col_dict[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = new_col_dict[col_list[j]]
        new_col = '%s_sub_%s' % (col_list[i], col_list[j])
        #new_col_list.append(new_col) 
        visited = set()
        c = a-b
        has_related = False
        for col, d in new_col_dict.items():
            rel = spearmanr(c, d)[0]
            if abs(rel) >= 0.45:
                has_related = True
                print 'spearman correlation %.3f breaking out! ' % rel
                break
        if not has_related:
            print 'new col %s added' % new_col
            new_col_dict[new_col] = c

for i in xrange(len(col_list)):
    a = new_col_dict[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = new_col_dict[col_list[j]]
        new_col = '%s_add_%s' % (col_list[i], col_list[j])
        #new_col_list.append(new_col) 
        visited = set()
        c = a+b
        has_related = False
        for col, d in new_col_dict.items():
            rel = spearmanr(c, d)[0]
            if abs(rel) >= 0.45:
                has_related = True
                print 'spearman correlation %.3f breaking out! ' % rel
                break
        if not has_related:
            print 'new col %s added' % new_col
            new_col_dict[new_col] = c

            
train.to_csv('train_feat_sele.csv', index = False)
del train
print 'processing test set....'
test = pd.read_csv('test.csv')
for i in xrange(len(col_list)):
    a = test[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = test[col_list[j]]
        test['%s_%s' % (col_list[i], col_list[j])] = a*b

test.to_csv('test_mult.csv', index = False)