import copy
import datetime
import numpy as np
import statsmodels.api as sm
import statsmodels as stats
from sklearn import linear_model


class RegressionResult(object):
    def __init__(self, **kwargs):
        self._params = None if 'params' not in kwargs else kwargs['params']
        self._tvalues = None if 'tvalues' not in kwargs else kwargs['tvalues']
        self._rsquared = None if 'rsquared' not in kwargs else kwargs['rsquared']

    def set_params(self, params):
        self._params = params

    def get_params(self):
        return self._params

    def set_tvalues(self, tvalues):
        self._tvalues = tvalues

    def get_tvalues(self):
        return self._tvalues if self._tvalues is not None else np.full(self._params.shape, np.nan)

    def set_rsquared(self, rsquared):
        self._rsquared = rsquared

    def get_rsquared(self):
        return self._rsquared if self._rsquared is not None else np.nan


class DayLearn(object):
    def __init__(self, xlist=None, y=None, valid=None, day_weight=None, symbol_weight=None):
        if xlist is None:
            xlist = []
        self.xlist = copy.deepcopy(xlist)
        self.y = y
        self.valid = valid
        self.day_weight = day_weight
        self.symbol_weight = symbol_weight
        self.panel_or_cross = 'p'
        self.l1_or_l2 = 'l2'
        self.need_const = True
        self.isLearned = False
        self.result = None
        self.dataLength = 0

    def clear(self):
        self.xlist = None
        self.y = None
        self.valid = None
        self.day_weight = None
        self.symbol_weight = None
        self.panel_or_cross = None
        self.l1_or_l2 = None
        self.need_const = None
        self.isLearned = None
        self.result = None
        self.dataLength = None

    def setXY(self, xy):
        x = xy[:-1, :, :]
        y = xy[-1, :, :]
        for i in range(x.shape[0]):
            self.xlist.append(x[i, :, :])
        self.y = y

    def addX(self, x):
        self.xlist.append(x)

    def checkData(self):
        if len(self.xlist) == 0:
            raise Exception('Please Add x first!')
        assert self.y is not None
        if self.valid is not None:
            assert self.valid.shape == self.y.shape
        else:
            self.valid = np.ones(self.y.shape)
        if self.day_weight is not None:
            assert len(self.day_weight) == self.y.shape[1]
        else:
            self.day_weight = np.ones(self.y.shape[1])
        if self.symbol_weight is not None:
            assert self.symbol_weight.shape == self.y.shape
        else:
            self.symbol_weight = np.ones(self.y.shape)

        for x in self.xlist:
            assert x.shape == self.y.shape
        assert self.valid.shape == self.y.shape
        assert self.day_weight.shape == (self.y.shape[1],)
        assert self.symbol_weight.shape == self.y.shape

    def learn(self, panel_or_cross='panel', l1_or_l2='l2', **kwargs):
        self.need_const = True if 'need_const' not in kwargs else kwargs['need_const']
        # regression parameter
        if panel_or_cross in ('panel', 'p', 'pool'):
            panel_or_cross = 'p'
        elif panel_or_cross in ('cross', 'c', 'cross-section'):
            panel_or_cross = 'c'
        else:
            raise Exception('Unknown type of panel_or_cross')

        if l1_or_l2 == 'l1':
            regFunc = self.l1_regression
        elif l1_or_l2 == 'l2':
            regFunc = self.l2_regression
        elif l1_or_l2 == 'ridge':
            regFunc = self.ridge_regression
        else:
            raise Exception('Unknown type of l1_or_l2')

        # get data
        self.checkData()
        xlist = self.xlist
        y = self.y
        valid = self.valid
        day_weight = self.day_weight
        symbol_weight = self.symbol_weight
        shapeall = y.shape
        if len(xlist[0].shape) <= 1:
            print 'Warning: Regression Data is Too Short (x shape is too small)!'
            self.isLearned = True
            self.result = None
            return
        if len(y.shape) <= 1:
            print 'Warning: Regression Data is Too Short (y shape is too small)!'
            self.isLearned = True
            self.result = None
            return

        # get reg data
        weight = symbol_weight * day_weight
        n_all = shapeall[0] * shapeall[1]
        reg_x = np.full((n_all, len(xlist)), np.nan)
        reg_y = y.T.reshape((n_all, 1))
        reg_w = weight.T.reshape((n_all, 1))
        reg_v = valid.T.reshape((n_all, 1))
        for i in range(len(xlist)):
            x = xlist[i]
            xnew = x.T.reshape((n_all, 1))
            reg_x[:, i] = xnew.ravel()

        # check reg data
        assert len(reg_x) == len(reg_y) == len(reg_w)
        if len(reg_x) <= 1000:
            print 'Warning: Regression Data is Too Short (reg_x length is %d)!' % (len(reg_x),)
            self.isLearned = True
            self.result = None
            return

        if panel_or_cross == 'p':
            reg_x_sum = np.sum(reg_x, axis=1).reshape((-1, 1))
            allexist = (~np.isnan(reg_x_sum)) & (~np.isnan(reg_y)) & (~np.isnan(reg_w)) & \
                       (~np.isnan(reg_v) & (reg_v > 0.5))
            ax = np.repeat(allexist, len(xlist)).reshape((-1, len(xlist)))
            reg_x = reg_x[ax].reshape(-1, len(xlist))
            reg_y = reg_y[allexist]
            reg_w = reg_w[allexist]
            assert len(reg_x) == len(reg_y) == len(reg_w)
            self.dataLength = len(reg_x)
            if len(reg_x) <= 100:
                print 'Warning: Regression Data is Too Short 2 (reg_x length is %d)!' % len(reg_x)
                self.isLearned = True
                self.result = None
                return
            print 'start regression:', datetime.datetime.now()
            self.result = regFunc(reg_y, reg_x, reg_w, **kwargs)
            print 'end regression:', datetime.datetime.now()
        elif panel_or_cross == 'c':
            self.result = []
            n_symbols = self.y.shape[0]
            n_days = self.y.shape[1]
            assert len(reg_x) == len(reg_y) == len(reg_v) == len(reg_w) == n_days * n_symbols
            for i in range(n_days):
                tmpreg_x1 = reg_x[i * n_symbols:(i + 1) * n_symbols, :]
                tmpreg_y1 = reg_y[i * n_symbols:(i + 1) * n_symbols]
                tmpreg_w1 = reg_w[i * n_symbols:(i + 1) * n_symbols]
                tmpreg_v1 = reg_v[i * n_symbols:(i + 1) * n_symbols]
                tmpreg_x1_sum = np.sum(tmpreg_x1, axis=1).reshape((-1, 1))
                allexist = (~np.isnan(tmpreg_x1_sum)) & (~np.isnan(tmpreg_y1)) & (~np.isnan(tmpreg_w1)) & \
                           (~np.isnan(tmpreg_v1) & (tmpreg_v1 > 0.5))
                ax = np.repeat(allexist, len(xlist)).reshape((-1, len(xlist)))
                tmpreg_x = tmpreg_x1[ax].reshape(-1, len(xlist))
                tmpreg_y = tmpreg_y1[allexist]
                tmpreg_w = tmpreg_w1[allexist]
                assert tmpreg_x.shape[0] == len(tmpreg_y) == len(tmpreg_w)
                assert tmpreg_x.shape[1] == len(xlist)
                if len(tmpreg_x) <= 100:
                    continue
                tmpresult = regFunc(tmpreg_y, tmpreg_x, tmpreg_w, **kwargs)
                self.result.append(tmpresult)
            if not self.result or len(self.result) < 100:
                self.result = None

        self.panel_or_cross = panel_or_cross
        self.l1_or_l2 = l1_or_l2
        self.isLearned = True

        return

    @staticmethod
    def l1_regression(y, x, w, **kwargs):
        need_const = True if 'need_const' not in kwargs else kwargs['need_const']
        if need_const:
            x = sm.add_constant(x)
        res = stats.regression.quantile_regression.QuantReg(y, x).fit()
        reg_res = RegressionResult(params=res.params,
                                   tvalues=res.tvalues,
                                   rsquared=res.rsquared)
        return reg_res

    @staticmethod
    def l2_regression(y, x, w, **kwargs):
        need_const = True if 'need_const' not in kwargs else kwargs['need_const']
        if need_const:
            x = sm.add_constant(x)
        res = sm.WLS(y, x, weights=w).fit()
        reg_res = RegressionResult(params=res.params,
                                   tvalues=res.tvalues,
                                   rsquared=res.rsquared)
        return reg_res

    @staticmethod
    def ridge_regression(y, x, w, **kwargs):
        need_const = True if 'need_const' not in kwargs else kwargs['need_const']
        alpha = 1.0 if 'alpha' not in kwargs else kwargs['alpha']
        rdr = linear_model.Ridge(alpha=alpha, fit_intercept=True)
        rdr.fit(x, y)
        params = np.concatenate([[rdr.intercept_], rdr.coef_]) if need_const else rdr.coef_
        reg_res = RegressionResult(params=params)
        return reg_res

    def getResult(self):
        if not self.isLearned:
            raise Exception('Please learn first!')
        return self.result

    def getCoef(self):
        if not self.isLearned:
            raise Exception('Please learn first!')
        if self.result is None:
            return np.full(len(self.xlist) + (1 if self.need_const else 0), np.nan)
        if self.panel_or_cross == 'p':
            para = self.result.get_params()
            if len(para) == len(self.xlist):
                para = np.concatenate(([0], para))
            return para
        elif self.panel_or_cross == 'c':
            params = np.zeros(len(self.xlist) + (1 if self.need_const else 0))
            n_results = len(self.result)
            for r in self.result:
                try:
                    params += r.get_params()
                except:
                    n_results -= 1
            if n_results == 0:
                return np.full(len(self.xlist) + (1 if self.need_const else 0), np.nan)
            params /= n_results
            return params

    def getTValue(self):
        if not self.isLearned:
            raise Exception('Please learn first!')
        if self.result is None:
            return np.full(len(self.xlist) + (1 if self.need_const else 0), np.nan)
        if self.panel_or_cross == 'p':
            return self.result.get_tvalues()
        elif self.panel_or_cross == 'c':
            if len(self.result) < 50:
                print 'Warning: result length is not long enough!'
            tvalueList = []
            for r in self.result:
                if len(r.get_tvalues()) == len(self.xlist) + (1 if self.need_const else 0):
                    tvalueList.append(r.get_tvalues())
            if len(tvalueList) == 0:
                return np.full(len(self.xlist) + (1 if self.need_const else 0), np.nan)
            tvalueArr = np.array(tvalueList)
            tmean = np.nanmean(tvalueArr, axis=0)
            tlen = np.sum(~np.isnan(tvalueList), axis=0)
            tstd = np.nanstd(tvalueArr, axis=0)
            tvalues = tmean / tstd * np.sqrt(tlen)
            return tvalues

    def getR2(self):
        if not self.isLearned:
            raise Exception('Please learn first!')
        if self.result is None:
            return np.nan
        if self.panel_or_cross == 'p':
            return self.result.get_rsquared()
        elif self.panel_or_cross == 'c':
            params = self.getCoef()
            xlist = self.xlist
            y = self.y
            valid = self.valid
            day_weight = self.day_weight
            symbol_weight = self.symbol_weight
            weight = symbol_weight * day_weight
            n_all = y.shape[0] * y.shape[1]
            reg_x = np.full((n_all, len(xlist)), np.nan)
            reg_y = y.T.reshape((n_all, 1))
            reg_w = weight.T.reshape((n_all, 1))
            reg_v = valid.T.reshape((n_all, 1))
            for i in range(len(xlist)):
                x = xlist[i]
                xnew = x.T.reshape((n_all, 1))
                reg_x[:, i] = xnew.ravel()
            reg_x_sum = np.sum(reg_x, axis=1).reshape((-1, 1))
            allexist = (~np.isnan(reg_x_sum)) & (~np.isnan(reg_y)) & (~np.isnan(reg_w)) & \
                       (~np.isnan(reg_v) & (reg_v > 0.5))
            ax = np.repeat(allexist, len(xlist)).reshape((-1, len(xlist)))
            reg_x = reg_x[ax].reshape(-1, len(xlist))
            reg_x_con = sm.add_constant(reg_x)
            reg_y = reg_y[allexist]
            reg_w = reg_w[allexist]
            reg_y_hat = np.dot(reg_x_con, params)
            rss = np.sum(reg_w * (reg_y - reg_y_hat) ** 2)
            tss = np.sum(reg_w * (reg_y ** 2))
            rsquared = 1 - rss / tss
            return rsquared

    def getDataLength(self):
        if not self.isLearned:
            raise Exception('Please learn first!')
        if self.result is None:
            return np.nan
        if self.panel_or_cross == 'p':
            return self.dataLength
        elif self.panel_or_cross == 'c':
            return len(self.result)
