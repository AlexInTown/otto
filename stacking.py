# -*- coding: utf-8 -*-
from sklearn.ensemble.base import BaseEnsemble
import numpy as np
from itertools import izip
#from ..grid_search import IterGrid
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import assert_all_finite
import logging

# TODO: capability to train base estimators seperately.
# TODO: built-in nested cross validation, re-using base classifiers,
# to pick best stacking method.
# TODO: access to best, vote, etc. after training.

__all__ = [
    "Stacking",
    "StackingFWL",
    'estimator_grid'
]

'''
def estimator_grid(*args):
    result = []
    pairs = izip(args[::2], args[1::2])
    for estimator, params in pairs:
        if len(params) == 0:
            result.append(estimator())
        else:
            for p in IterGrid(params):
                result.append(estimator(**p))
    return result
'''

class MRLR(ClassifierMixin):
    '''Converts a multi-class classification task into a set of
    indicator regression tasks.
    Ting, K.M., Witten, I.H.: Issues in stacked generalization.
    Journal of Artificial Intelligence Research 10,271–289 (1999)
    '''
    def __init__(self, regressor, stackingc, **kwargs):
        self.estimator_ = regressor
        self.estimator_args_ = kwargs
        self.stackingc_ = stackingc

    def _get_subdata(self, X):
        '''Returns subsets of the data, one for each class. Assumes the
        columns of X are striped in order.
        e.g. if n_classes_ == 3, then returns (X[:, 0::3], X[:, 1::3],
        X[:, 2::3])
        '''
        if not self.stackingc_:
            return [X, ] * self.n_classes_

        result = []
        for i in range(self.n_classes_):
            slc = (slice(None), slice(i, None, self.n_classes_))
            result.append(X[slc])
        return result

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.estimators_ = []

        X_subs = self._get_subdata(X)

        for i in range(self.n_classes_):
            e = self.estimator_(**self.estimator_args_)
            y_i = np.array(list(j == i for j in y))
            X_i = X_subs[i]
            e.fit(X_i, y_i)
            self.estimators_.append(e)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        proba = []

        X_subs = self._get_subdata(X)

        for i in range(self.n_classes_):
            e = self.estimators_[i]
            X_i = X_subs[i]
            pred = e.predict(X_i).reshape(-1, 1)
            proba.append(pred)
        proba = np.hstack(proba)

        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        assert_all_finite(proba)

        return proba


class Stacking(BaseEnsemble):
    '''
    Implements stacking.
    David H. Wolpert (1992). Stacked generalization. Neural Networks,
    5:241-259, Pergamon Press.
    Params
    ------
    + meta_estimator : may be one of "best", "vote", "average", or any
      classifier or regressor constructor.
    + estimators : an iterable of estimators; each must support
      predict_proba()
    + cv : a cross validation object. Level 0 estimators are trained
      on the training folds, then the meta estimator is trained on the
      testing folds.
    + stackingc : whether to use StackingC or not. For more
      information, this paper:
          Seewald A.K.: How to Make Stacking Better and Faster While
          Also Taking Care of an Unknown Weakness, in Sammut C.,
          Hoffmann A. (eds.), Proceedings of the Nineteenth
          International Conference on Machine Learning (ICML 2002),
          Morgan Kaufmann Publishers, pp.554-561, 2002.
    + kwargs : arguments passed to instantiate meta_estimator.
    '''

    # TODO: support different features for each estimator
    # TODO: support "best", "vote", and "average" for already trained
    # model.
    # TODO: allow saving of estimators, so they need not be retrained
    # when trying new stacking methods.

    def __init__(self, meta_estimator, estimators,
                 cv, stackingc=True,
                 **kwargs):
        self.estimators_ = estimators
        self.n_estimators_ = len(estimators)
        self.cv_ = cv
        self.stackingc_ = stackingc

        if stackingc and not issubclass(meta_estimator, RegressorMixin):
            raise Exception('StackingC only works with a regressor.')

        if isinstance(meta_estimator, str):
            if meta_estimator not in ('best',
                                      'average',
                                      'vote'):
                raise Exception('invalid meta estimator: {0}'.format(meta_estimator))
            raise Exception('"{0}" meta estimator not implemented'.format(meta_estimator))
        elif issubclass(meta_estimator, ClassifierMixin):
            self.meta_estimator_ = meta_estimator(**kwargs)
        elif issubclass(meta_estimator, RegressorMixin):
            self.meta_estimator_ = MRLR(meta_estimator, stackingc, **kwargs)
        else:
            raise Exception('invalid meta estimator: {0}'.format(meta_estimator))

    def _make_meta(self, X):
        rows = []
        for e in self.estimators_:
            proba = e.predict_proba(X)
            assert_all_finite(proba)
            rows.append(proba)
        return np.hstack(rows)

    def fit(self, X, y):
        # Build meta data
        X_meta = []
        y_meta = []

        for a, b in self.cv_:
            logging.info('new stacking fold')
            X_a, X_b = X[a], X[b]
            y_a, y_b = y[a], y[b]

            for e in self.estimators_:
                logging.info('stacking training level-0 estimator {0}'.format(e))
                e.fit(X_a, y_a)

            proba = self._make_meta(X_b)
            #print proba.shape
            X_meta.append(proba)
            y_meta.append(y_b)

        X_meta = np.vstack(X_meta)
        if y_meta[0].ndim == 1:
            y_meta = np.hstack(y_meta)
        else:
            y_meta = np.vstack(y_meta)

        # train meta estimator
        logging.info('training meta estimator')
        self.meta_estimator_.fit(X_meta, y_meta)

        # re-train estimators on full data
        logging.info('re-training estimators on full data')
        for e in self.estimators_:
            e.fit(X, y)

    def predict(self, X):
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict(X_meta)

    def predict_proba(self, X):
        X_meta = self._make_meta(X)
        return self.meta_estimator_.predict_proba(X_meta)


class StackingFWL(Stacking):
    '''
    Implements Feature-Weighted Linear Stacking.
    Sill, J. and Takacs, G. and Mackey, L. and Lin, D.:
    Feature-weighted linear stacking. Arxiv preprint. 2009.
    '''
    pass