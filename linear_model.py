__author__ = 'rxfan'

import random
import copy
from abc import abstractmethod

import numpy as np
from sklearn import linear_model
import statsmodels.api as sm


class AbstractLinearModel(object):
    def __init__(self, y, x, **args):
        self.y = y
        self.x = x
        self.args = args

        self._is_fitted = False
        self._model = None
        self._coefficient = None
        self._t_value = None
        self._r_squared = None

    @classmethod
    @abstractmethod
    def print_default_parameter(cls):
        pass

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def get_coefficient(self):
        raise NotImplementedError()

    @abstractmethod
    def get_t_value(self):
        raise NotImplementedError()

    @abstractmethod
    def get_r_squared(self):
        raise NotImplementedError()


class OrdinaryLeastSquares(AbstractLinearModel):
    _default_parameter = dict(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1, sample_weight=None)

    def __init__(self, y, x, **args):
        super(OrdinaryLeastSquares, self).__init__(y, x, **args)
        self.para = copy.deepcopy(self._default_parameter)
        self.para.update(args)
        fit_intercept = self.para['fit_intercept']
        normalize = self.para['normalize']
        copy_X = self.para['copy_X']
        n_jobs = self.para['n_jobs']
        self._model = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize,
                                                    copy_X=copy_X, n_jobs=n_jobs)

    @classmethod
    def print_default_parameter(cls):
        for para in cls._default_parameter:
            print para, ':', cls._default_parameter[para]
        print '''
****************************
for more information:
from sklearn import linear_model
linear_model.LinearRegression?
        '''

    def fit(self):
        sample_weight = self.para['sample_weight']
        self._model.fit(self.x, self.y, sample_weight=sample_weight)
        self._is_fitted = True

    def get_coefficient(self):
        assert self._is_fitted
        self._coefficient = self._model.intercept_
        return self._coefficient

    def get_r_squared(self):
        assert self._is_fitted
        return None

    def get_t_value(self):
        assert self._is_fitted
        return None


class L2Regression(AbstractLinearModel):
    _default_parameter = dict()

    def __init__(self, y, x, **args):
        super(L2Regression, self).__init__(y, x, **args)


    @classmethod
    def print_default_parameter(cls):
        pass

    def get_coefficient(self):
        pass

    def get_r_squared(self):
        pass

    def get_t_value(self):
        pass

    def fit(self):
        pass


class L1Regression(AbstractLinearModel):
    _default_parameter = dict()

    def __init__(self, y, x, **args):
        super(L1Regression, self).__init__(y, x, **args)


    @classmethod
    def print_default_parameter(cls):
        pass

    def get_coefficient(self):
        pass

    def get_r_squared(self):
        pass

    def get_t_value(self):
        pass

    def fit(self):
        pass
    

class WeightedLeastSquares(AbstractLinearModel):
    pass
    # TODO


class LASSO(AbstractLinearModel):
    pass


class RidgeRegression(AbstractLinearModel):
    pass


class ElasticNet(AbstractLinearModel):
    pass


class LinearModelFactory(object):
    _method_dict = {'OLS': OrdinaryLeastSquares,
                    'L1': L1Regression,
                    # 'WLS': WeightedLeastSquares,
                    # 'LASSO': LASSO,
                    # 'Ridge': RidgeRegression,
                    # 'Elastic': ElasticNet,
                    }

    def __init__(self, y=None, x=None, method=None, **kwargs):
        self._y = y
        self._x = x
        self._method = method
        self._args = kwargs

    def get_available_methods(self):
        return self._method_dict.keys()

    def get_parameter_help(self, method):
        return self._method_dict[method].print_default_parameter()

    def _set_method(self, method):
        assert method in self._method_dict
        self._method = method

    def create_model(self, y=None, x=None, method=None, **kwargs):
        self._y = y if y else self._y
        self._x = x if x else self._x
        self._args = kwargs if kwargs else self._args
        if method:
            self._set_method(method)
        assert self._method
        return self._method_dict[self._method](self._y, self._x, **self._args)


# if __name__ == '__main__':
#     test_x = range(100)
#     test_y = [random.normalvariate(0, 1) for _ in range(100)]
#
#     factory = LinearModelFactory()
#     for method in factory.get_available_methods():
#         print method
#     factory.get_parameter_help('OLS')
#     model = factory.create_model(test_y, test_x, method='OLS')
