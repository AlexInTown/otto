import numpy as np
import pandas as pd
from sklearn import feature_extraction, preprocessing, cluster
from scipy.stats import spearmanr

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
#col_list = ['feat_%d'%i for i in [85, 66, 14, 24, 39, 25, 13, 59, 10, 33]]
new_col_list = []
new_col_dict = {}
k = 0
for i in xrange(len(col_list)):
    a = train[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = train[col_list[j]]
        #train['%s_%s' % (col_list[i], col_list[j])] = a*b
        new_col_list.append('%s_%s' % (col_list[i], col_list[j])) 
        new_col_dict.append(a*b)
        k += 1

train.to_csv('train_mult.csv', index = False)
del train
print 'processing test set....'
test = pd.read_csv('test.csv')
for i in xrange(len(col_list)):
    a = test[col_list[i]]
    for j in xrange(i+1, len(col_list)):
        b = test[col_list[j]]
        test['%s_%s' % (col_list[i], col_list[j])] = a*b

test.to_csv('test_mult.csv', index = False)