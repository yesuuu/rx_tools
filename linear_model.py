import copy


class LinearModelResult(object):
    def __init__(self, params=None, rsquared=None, tvalue=None):
        self._params = params
        self._rsquared = rsquared
        self._tvalue = tvalue

    def get_params(self):
        return copy.deepcopy(self._params)

    def set_params(self, params):
        self._params = params

    def get_rsquared(self):
        return self._rsquared

    def set_rsquared(self, rsquared):
        self._rsquared = rsquared

    def get_tvalue(self):
        return copy.deepcopy(self._tvalue)

    def set_tvalue(self, tvalue):
        self._tvalue = tvalue



