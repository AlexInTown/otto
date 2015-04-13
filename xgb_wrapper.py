import xgboost as xgb
from sklearn.base import ClassifierMixin
class XgbWrapper (ClassifierMixin):
    """
    Wrapper class for xgboost.Booster 
    """
    def __init__(self, param, num_round = 200, watchlist = ()):
        """
        """
        self.param = param
        self.num_round = num_round
        self.watchlist = watchlist

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.bst = xgb.Booster(self.param, [dtrain] + [d[0] for d in self.watchlist])
        for i in range(self.num_round):
            self.bst.update(dtrain, i)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X, label=None)
        preds = self.bst.predict(dtest)
        #print "preds shape ", preds.shape
        return preds