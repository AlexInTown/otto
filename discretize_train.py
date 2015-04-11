# coding=UTF-8
"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, grid_search

# import data
train = pd.read_csv('discretize_train.csv')
# test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
#print labels
train = train.drop(['id','target'], axis=1)
#test = test.drop(['id'], axis=1)

# transform counts to TFIDF features
# tfidf = feature_extraction.text.TfidfTransformer()
# train = tfidf.fit_transform(train).toarray()
# test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
# clf = ensemble.RandomForestClassifier(n_jobs=4, n_estimators=400)
# clf = ensemble.GradientBoostingClassifier(n_estimators=100, verbose=True)
clf = linear_model.LogisticRegression('l2', C=3.0)
clf.fit(train, labels)

num_parts = 10
# create submission file
final_preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
for i in xrange(num_parts):
    test = pd.read_csv('discretize_test_part%d.csv' % i)
    test = test.drop(['id'], axis=1)
    # predict on test set
    preds = clf.predict_proba(test)
    preds = pd.DataFrame(preds, index=np.arange(preds.shape[0]), columns=sample.columns[1:])
    final_preds.append(preds);
preds.to_csv('benchmark.csv', index_label='id')