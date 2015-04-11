# coding=UTF-8
"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
#print labels
train = train.drop(['id','target'], axis=1)
test = test.drop(['id'], axis=1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# train a random forest classifier
clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
# clf = ensemble.GradientBoostingClassifier(n_estimators=100, verbose=True)
# m_cv = cross_validation.KFold(len(labels), n_folds=5,shuffle=True, indices=True)
m_param_grid={'n_estimators': (100, 50), 'max_leaf_nodes': (None, 50, 80, 100), 'criterion': ('gini', 'entropy')}
m_scorer = metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True)
gsearch = grid_search.GridSearchCV(cv=5, estimator=clf, param_grid=m_param_grid, n_jobs=1, refit=True, scoring= m_scorer, verbose=2)

gsearch.fit(train, labels)

print gsearch.grid_scores_
print gsearch.best_estimator_
print gsearch.best_score_
print gsearch.best_params_
print gsearch.scorer_

# predict on test set
clf = gsearch.best_estimator_
preds = clf.predict_proba(test)

# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('new_score.csv', index_label='id')