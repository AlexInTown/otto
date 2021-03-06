import pandas as pd
import pandas as pd
import numpy as np
import sys
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation, metrics, linear_model, decomposition
import xgboost as xgb
reload(xgb)
import cPickle as cp
import otto_utils


class Level2Classification:
    def __init__(self, train, labels, cls_sets):
        self.cls_sets = cls_sets
        self.preprocess_train(train, labels)

    def preprocess_train(self, train, labels):
        self.lbl_enc = preprocessing.LabelEncoder()
        self.labels = self.lbl_enc.fit_transform(labels)  # change raw labels to integer labels
        self.nclass = len(self.lbl_enc.classes_)
        self.raw_labels = labels
        i = self.nclass 
        self.l1_train = train
        self.l1_labels = labels.copy()
        self.l2_train = []
        self.l2_labels = []
        self.class_father = {}
        for ss in self.cls_sets:
            l2_train_temp = []
            l2_labels_temp = []
            for class_id in ss:
                l2_train_temp.append(train[labels==class_id])    # separate level2 classes
                l2_labels_temp.append(labels[labels==class_id])
                new_l1_class_id = "L1_Class_ID_%d"%i               # new level1 class ID (starts from nclass)
                self.class_father[class_id] = new_l1_class_id
                self.l1_labels[labels==class_id] = new_l1_class_id
            self.l2_train.append(pd.concat(l2_train_temp))
            self.l2_labels.append(np.concatenate(l2_labels_temp))
            i += 1
        self.l1_lbl_enc = preprocessing.LabelEncoder()
        self.l1_labels = self.l1_lbl_enc.fit_transform(self.l1_labels)
        self.l2_lbl_encs = []
        l2_labels_temp = []
        for i in xrange(len(self.cls_sets)):
            self.l2_lbl_encs.append(preprocessing.LabelEncoder())
            l2_labels_temp.append(self.l2_lbl_encs[-1].fit_transform(self.l2_labels[i]))
        self.l2_labels = l2_labels_temp


    def train_level1(self, select_param = False):
        self.l1_param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
        other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
        other['num_class'] = len(self.l1_lbl_enc.classes_)
        self.l1_num_round = 400
        if select_param:
            results = otto_utils.select_xgb_params(self.l1_train, self.l1_labels, depth_range=(7,12))
            self.l1_results = results
            print results[0]
            self.l1_num_round  = results[0][1]
            param = results[0][2]
        dtrain = xgb.DMatrix(self.l1_train, label=self.l1_labels)
        full_param = other.copy()
        full_param.update(self.l1_param) 
        bst,loss,ntree = xgb.train(full_param, dtrain, self.l1_num_round, [(dtrain,'train')])
        self.l1_model = bst
        pass
 
    def train_level2(self, select_param = False):
        self.l2_models = []
        self.l2_params = []
        self.l2_num_rounds = []
        self.l2_results = []
        for i in xrange(len(self.cls_sets)):
            param = {'bst:max_depth':10, 'bst:min_child_weight': 4, 'bst:subsample': 0.5, 'bst:colsample_bytree':0.8,  'bst:eta':0.05}
            other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
            other['num_class'] = len(self.cls_sets[i])
            num_round = 400
            if select_param:
                results = otto_utils.select_xgb_params(self.l2_train[i], self.l2_labels[i], depth_range=(8, 13))
                self.l2_results.append(results)
                print results[0]
                num_round = results[0][1]
                param = results[0][2]
            dtrain = xgb.DMatrix(self.l2_train[i], self.l2_labels[i])
            self.l2_params.append(param)
            self.l2_num_rounds.append(num_round)
            full_param = other.copy()
            full_param.update(param) 
            bst,loss,ntree = xgb.train(full_param, dtrain, num_round , [(dtrain,'train')])
            self.l2_models.append(bst)
        pass

    def predict_level1(self, test):
        dtest = xgb.DMatrix(test)
        return self.l1_model.predict(dtest)

    def predict_level2(self, test):
        dtest = xgb.DMatrix(test)
        res = []
        for i in xrange(len(self.cls_sets)):
            res.append(self.l2_models[i].predict(dtest))
        return res

    def train(self, select_param = False):
        self.train_level1(select_param = select_param)
        self.train_level2(select_param = select_param)

    def predict(self, test):
        preds = self.predict_level1(test)
        res = {}
        for i in xrange(preds.shape[1]):
            res[self.l1_lbl_enc.classes_[i]] = preds[:,i]
        l2_preds = self.predict_level2(test)
        for i in xrange(len(l2_preds)):
            pred = l2_preds[i]
            for j in xrange(pred.shape[1]):
                raw_id = self.l2_lbl_encs[i].classes_[j]
                res[raw_id] = pred[:,j] * res[self.class_father[raw_id]]
        df = pd.DataFrame(index = np.arange(1, len(preds)+1))
        for raw_id in self.lbl_enc.classes_:
            df[raw_id] = res[raw_id]
        return df
        
    def load_param_select_results(self):
        self.l1_results = cp.load(open('hierarchy_param_level1_sel_100.pkl', 'rb'))
        self.l2_results = []
        self.l2_results.append(cp.load(open('hierarchy_param_level2_sel_100_0.pkl', 'rb')))
        self.l2_results.append(cp.load(open('hierarchy_param_level2_sel_100_1.pkl', 'rb')))

    def avg_predict(self, test, num_model = 15):
        dtest = xgb.DMatrix(test)
        l1_preds = None
        other = {'silent':1, 'objective':'multi:softprob', 'num_class':9, 'nthread': 4, 'eval_metric': 'mlogloss', 'seed':0}
        other['num_class'] = len(self.l1_lbl_enc.classes_)
        dtrain = xgb.DMatrix(self.l1_train, label=self.l1_labels)
        print 'Averaging selected parameters for level1 model.....'
        for loss, ntree, param in self.l1_results[:num_model]:
            full_param = other.copy()
            full_param.update(param) 
            bst,loss,ntree = xgb.train(full_param, dtrain, ntree , [])
            preds = bst.predict(dtest)
            if l1_preds is None:
                l1_preds = preds
            else:
                l1_preds += preds
        l1_preds /= num_model
        self.l1_preds = l1_preds
        print 'Averaging selected parameters for level2 model.....'
        self.l2_preds = []
        for i in xrange(len(self.cls_sets)):
            print ' runing level2 set %d ...' % i
            dtrain = xgb.DMatrix(self.l2_train[i], label=self.l2_labels[i])
            other['num_class'] = len(self.l2_lbl_encs[i].classes_)
            self.l2_preds.append(np.zeros((len(test), other['num_class'])))
            for loss, ntree, param in self.l2_results[i][:num_model]:
                full_param = other.copy()
                full_param.update(param) 
                bst, loss, ntree = xgb.train(full_param, dtrain, ntree, [])
                preds = bst.predict(dtest)
                self.l2_preds[i] += preds
        for i in xrange(len(self.cls_sets)):
            self.l2_preds[i] /= num_model
        preds = self.l1_preds
        res = {}
        for i in xrange(preds.shape[1]):
            res[self.l1_lbl_enc.classes_[i]] = preds[:,i]
        for i in xrange(len(self.l2_preds)):
            pred = self.l2_preds[i]
            for j in xrange(pred.shape[1]):
                raw_id = self.l2_lbl_encs[i].classes_[j]
                res[raw_id] = pred[:,j] * res[self.class_father[raw_id]]
        df = pd.DataFrame(index = np.arange(1, len(preds)+1))
        for raw_id in self.lbl_enc.classes_:
            df[raw_id] = res[raw_id]
        return df
