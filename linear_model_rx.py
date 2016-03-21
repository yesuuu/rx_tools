__author__ = 'rxfan'

import random
import copy
from abc import abstractmethod

import numpy as np
from sklearn import linear_model


class AbstractLinearModel(object):
    def __init__(self, y=None, x=None, **kwargs):
        self.y = y
        self.x = x
        self.args = kwargs

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
    _default_parameter = dict(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

    def __init__(self, y, x, **kwargs):
        super(OrdinaryLeastSquares).__init__(y, x, **kwargs)
        self.para = copy.deepcopy(self._default_parameter).update(kwargs)
        fit_intercept = self.para['fit_intercept']
        normalize = self.para['normalize']
        copy_X = self.para['copy_X']
        n_jobs = self.para['n_jobs']
        self._model = linear_model.LinearRegression

    @classmethod
    def print_default_parameter(cls):
        for para in cls._default_parameter:
            print para, cls._default_parameter[para]
        print '''from sklearn import linear_model
        call linear_model.LinearRegression for more information.
        '''

    def fit(self):
        para = copy.deepcopy(self._default_parameter)
        para.update(self.args)

        self.ols_model = linear_model.L


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
                    'WLS': WeightedLeastSquares,
                    'LASSO': LASSO,
                    'Ridge': RidgeRegression,
                    'Elastic': ElasticNet, }

    def __init__(self, y=None, x=None, method=None, **kwargs):
        self._y = y
        self._x = x
        self._method = method
        self._args = kwargs

    def get_available_methods(self):
        return self._method_dict.keys()

    def get_parameter_help(self, method):
        if method not in self._method_dict:
            raise Exception('Unknown method!')
        elif method == 'OLS':
            print '# TODO'
            return
        elif method == 'WLS':
            print '# TODO'

    def _set_method(self, method):
        if method not in self._method_dict:
            raise Exception('Unknown method!')
        else:
            self._method = method

    def create_model(self, y=None, x=None, method=None, **kwargs):
        self._y = y if y else self._y
        self._x = x if x else self._x
        self._args = kwargs if kwargs else self._args
        if method:
            self._set_method(method)
        assert self._method
        return self._method_dict[self._method](self._y, self._x, self._args)


if __name__ == '__main__':
    test_x = range(100)
    test_y = [random.normalvariate(0, 1) for _ in range(100)]

    factory = LinearModelFactory()
    for method in factory.get_available_methods():
        print method
    factory.get_parameter_help('OLS')
    model1 = factory.create_model(y=test_y, x=test_x, method='OLS')
