import numpy as np

class StackingLinearRegression:

    def __init__(self, lr, weight_decay, max_epoch, tol): 
        '''
        Linear Regression for stacking other basic classifiers
        Input: preds from other classifiers and true labels(one hot representation)
        Output: model weighted predictions
        Init Param:
            lr: Base learning rate 
            weight_decay: regularization term
            max_epoch: max iteration times
            tol:  define when to stop to get the best error ( error - best_error > tol)
        '''
        self.lr = lr
        self.decay = weight_decay
        self.iter = 0
        self.max_epoch = max_epoch
        self.tol = tol
        pass
    def iterate(self, data, label):
        self.pred.fill(0)
        self.dw.fill(0)
        self.diff = self.cal_error(data, label)
        for j in xrange(self.n):
            self.dw[j] = np.sum(np.sum(self.diff * data[j],axis=1))
        '''
        for i in xrange(self.m):
            for kk in xrange(self.k):
                for j in xrange(self.n):
                    #print self.dw[j], self.diff[i][kk], data[j][i][kk]
                    self.dw[j] += self.diff[i][kk] * data[j][i][kk]
                    #try:
                    #    self.dw[j] += self.diff[i][kk] * data[j][i][kk]
                    #except:
                    #    print i, kk, j
                    #    raise
        '''
        self.dw /= self.m*self.k
        self.iter += 1
     
    
    def cal_error(self, data, label):
        ''' 
            Be careful to the self.preds (shape(num_train_cases, num_classes)) when used to store the validation or test preds.
            There should be num_valid_cases <= num_train_cases and num_test_cases.
            return diff(shape(num_cases, num_classes))
        '''     
        diff = np.zeros(label.shape)
        # for each training case
        for i in xrange(label.shape[0]):
            # out preds from different models
            for j in xrange(self.n):
                self.pred[i] += self.w[j]*data[j][i]
            diff[i] = self.pred[i] - label[i]
        #print "diff.shape",diff.shape
        return diff
    
    def update(self):
        self.w += -self.lr *(self.dw - self.w * self.decay)
        
    def fit(self, data, label):
        self.n = len(data)       # number of models
        if self.n == 0:
            raise "Invalid training data length ", self.n
            
        # y[i][k] = sigma_j(w[j]*x[j][i][k])
        # E[i] = 0.5*sigma_k(y[i]-t[i])^2
        # E = 0.5 sigma_k sigma_i(y[i][k] - t[i][k]) ^2
        # E'(w[j]) = sigma_i sigma_k((y[i][k]-t[i][k])*x[j][i][k])
        # dw[j] = E'(w[j])
        # w[j] = -lr(dw[j] - decay*w[j])
        self.pred = np.zeros(label.shape)
        self.dw = np.zeros(self.n)
        self.w = np.ones(self.n)
        self.w /= self.n
        self.m = len(label)      # number of cases
        self.k = label.shape[1]  # number of classes
        best_error = None
        while self.iter < self.max_epoch:
            self.iterate(data, label)
            self.update()
            error = np.sum(np.sum(self.diff*self.diff,axis = 1)) / self.m
            if best_error == None or best_error > error:
                best_error = error 
            elif error - best_error > self.tol:
                print 'Breaking out at iteration %d, with error = %.6f and best_error = %.6f' %(self.iter, error, best_error)
                break
            elif error > best_error:
                self.lr *= 0.9
            print ' - Iter %d, error = %.6f, best_error=%.6f' % (self.iter, error, best_error)
            
    def predict(self, to_pred):
        if self.n != len(to_pred) or self.n == 0:
            raise "Invalid testing data length ", len(to_pred)
        preds = np.zeros(to_pred[0].shape)
        for i in xrange(len(to_pred[0])):
            for j in xrange(self.n):
                preds[i] += to_pred[j][i] * self.w[j]
            preds[i] /= self.n
        #print preds.shape
        return preds