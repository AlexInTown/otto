import numpy as np
import pandas as pd
from sklearn import feature_extraction, preprocessing, cluster

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop ids and get labels
labels = train.target.values
#print labels
#train = train.drop(['id','target'], axis=1)
#test = test.drop(['id'], axis=1)

# encode labels 
#lbl_enc = preprocessing.LabelEncoder()
#labels = lbl_enc.fit_transform(labels)

df = pd.DataFrame(index=np.arange(labels.size))
df['id'] = train['id']

clusters = {}
print 'Discretized train set'
for column in list(train.columns.values)[1:-1]:
    print '---Processing column %s----' % column
    feat = train[column]
    feat = np.reshape(feat, (-1, 1))
    #print feat.shape
    km = cluster.KMeans(10)
    km.fit(feat)
    clusters[column] = km
    new_feat = km.predict(feat)
    #print new_feat.shape
    new_feat = np.reshape(new_feat, (-1,1))
    #print new_feat
    oneHotEnc = preprocessing.OneHotEncoder()
    oneHotEnc.fit(new_feat)
    new_feats =oneHotEnc.fit_transform(new_feat)
    #print new_feats.shape
    new_feats = new_feats.transpose()  # shape=(num_rows, new_num_cols)
    #print new_feats.shape
    rows,cols = new_feats.shape        # shape=(new_num_cols, num_rows)
    
    for i in xrange(rows):
        df[column+'_'+str(i)] = new_feats[i].toarray()[0]
df['target'] = train['target']
#df.set_index(['id'])
df.to_csv('discretize_train.csv', index=False)


# df = pd.DataFrame(index=np.arange(test.shape[0]))
# df['id'] = test['id']
# for column in list(train.columns.values)[1:]:
    # print '---Processing column %s----' % column
    # km = clusters[column]
    # test_feat = test[column]
    # test_feat = np.reshape(test_feat, (-1, 1))
    # new_feat = km.predict(test_feat)
    # new_feat = np.reshape(new_feat, (-1,1))
    # oneHotEnc = preprocessing.OneHotEncoder()
    # oneHotEnc.fit(new_feat)
    # new_feats =oneHotEnc.fit_transform(new_feat)
    # new_feats = new_feats.transpose()  # shape=(num_rows, new_num_cols)
    # rows,cols = new_feats.shape        # shape=(new_num_cols, num_rows)
    # for i in xrange(rows):
        # df[column+'_'+str(i)] = new_feats[i].toarray()[0]
# df.to_csv('discretize_test.csv')

print 'Discretized test set'
test_size = test.shape[0]
num_parts = 10
avg_size = test_size / num_parts
print 'Total number of test set: %d' %test_size
print 'Average number of each part: %d' %avg_size
cur_start = 0
for i in xrange(num_parts):
    print '------Processing part %d------' % i
    if i == num_parts-1:
        part_df = test[cur_start:]
    else:
        part_df = test[cur_start:cur_start + avg_size]
    part_df['id'] = test['id'][cur_start: cur_start + avg_size]
    df = pd.DataFrame(index=np.arange(part_df.shape[0]))
    df['id'] = part_df['id']
    #df.set_index(['id'])
    for column in list(test.columns.values)[1:]:
        print '.....Processing column %s....' % column
        km = clusters[column]
        test_feat = part_df[column]
        test_feat = np.reshape(test_feat, (-1, 1))
        new_feat = km.predict(test_feat)
        new_feat = np.reshape(new_feat, (-1,1))
        oneHotEnc = preprocessing.OneHotEncoder()
        oneHotEnc.fit(new_feat)
        new_feats =oneHotEnc.fit_transform(new_feat)
        new_feats = new_feats.transpose()  # shape=(num_rows, new_num_cols)
        rows,cols = new_feats.shape        # shape=(new_num_cols, num_rows)
        for j in xrange(rows):
            df[column+'_'+str(j)] = new_feats[j].toarray()[0]
    df.to_csv('discretize_test_part%d.csv'%i, index=False)
    cur_start += avg_size