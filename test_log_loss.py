import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model

# import data
#train = pd.read_csv('discretize_train.csv')
train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')
#sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
#print labels
train = train.drop(['id','target'], axis=1)

# transform counts to TFIDF features
#tfidf = feature_extraction.text.TfidfTransformer()
#train = tfidf.fit_transform(train).toarray()


# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# split trainset and testset
train, test, train_labels, test_labels = cross_validation.train_test_split(train, labels, test_size=0.3)

# train a random forest classifier
#clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=200, criterion='gini')
#clf = ensemble.GradientBoostingClassifier(n_estimators=200, verbose=True)
# clf = linear_model.LogisticRegression('l2', C=3.0)
# clf = ensemble.RandomForestClassifier(n_jobs=4, max_features=0.2, n_estimators=100)
clf = ensemble.GradientBoostingClassifier(n_estimators=200, verbose=True, max_depth=8)
clf.fit(train, train_labels)

# predict on test set
preds = clf.predict_proba(test)
oneHotEnc = preprocessing.OneHotEncoder()
test_labels = np.reshape(test_labels, (test_labels.size, 1))
oneHotEnc.fit(test_labels)
truth=oneHotEnc.fit_transform(test_labels)
print metrics.log_loss(truth, preds)