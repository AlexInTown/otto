import numpy as np
import pandas as pd
from sklearn import feature_extraction, preprocessing, cluster, cross_validation

# load train and test
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# get the labels
labels = train.target.values

# one-hot form of labels
lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(labels)
labels = lbl_enc.fit_transform(labels)

oneHotEnc = preprocessing.OneHotEncoder()
labels=oneHotEnc.fit_transform(labels.reshape((-1,1)))
labels=labels.toarray()

# remove un-used labels 
train = train.drop(['id', 'target'], axis=1)
test = test.drop(['id'], axis=1)

# standardize the data for DL training
scaler = preprocessing.StandardScaler(copy=False)
train = scaler.fit_transform(train)
test = scaler.transform(test)

# split the raw dataset into train and validation dataset
#train, validation, train_labels, valid_labels = cross_validation.train_test_split(train, labels, test_size=19878)

print 'train shape ', train.shape
#print 'valid shape ', validation.shape
print 'test shape ', test.shape
#print 'train_labels shape ', train_labels.shape
#print 'test_labels shape ', valid_labels.shape

# save to csv files
#np.savetxt('train_for_matlab.csv', train, delimiter=',', fmt='%.10e')
#np.savetxt('valid_for_matlab.csv', validation, delimiter=',', fmt='%.10e')
#np.savetxt('test_for_matlab.csv', test, delimiter=',', fmt='%.10e')
#np.savetxt('train_labels_for_matlab.csv', train_labels, delimiter=',', fmt='%.10e')
#np.savetxt('valid_labels_for_matlab.csv', valid_labels, delimiter=',', fmt='%.10e')
np.savetxt('full_for_matlab.csv', train, delimiter=',', fmt='%.10e')
np.savetxt('labels_full_for_matlab.csv', labels, delimiter=',', fmt='%.10e')
