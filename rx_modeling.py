import sys
import os
import time
import datetime
import re
import random
import subprocess
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn

matplotlib.use("Qt4Agg")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 15)


class RxModeling(object):

    class SysRelated(object):

        @staticmethod
        def batchRename(tarFolder, files, tarFiles):
            for i, j in zip(files, tarFiles):
                os.system('mv %s %s' % (os.path.join(tarFolder, i), os.path.join(tarFolder, j)))

        @staticmethod
        def batch_run(run_list, max_batch=1, wait_time=0, is_print=True, omp_num_threads=1):
            """
            input:

            max_batch: batches run at same time
            wait_time: when one run,

            """
            run_list = run_list[:]
            runnings = {}

            while run_list or runnings:
                for f in runnings.keys():
                    if runnings[f][0].poll() is not None:
                        time_diff = datetime.datetime.now() - runnings[f][1]
                        if is_print:
                            print '\n[BatchRun process end] %s' \
                                  '\n[BatchRun process end] use_time: %s' % (f, time_diff)
                            if len(run_list) == 0:
                                print '[BatchRun] %d left' % (len(run_list) + len(runnings) - 1,)
                        runnings.pop(f)
                if (len(runnings) < max_batch) and run_list:
                    run_now = run_list.pop(0)
                    f = subprocess.Popen("OMP_NUM_THREADS=%d %s" % (omp_num_threads, run_now), shell=True)
                    now = datetime.datetime.now()
                    if is_print:
                        print ('\n[BatchRun %d] OMP_NUM_THREADS=%d %s' % (f.pid, omp_num_threads, run_now))
                        # print '[BatchRun] time:', now
                    runnings[run_now] = [f, now]
                    time.sleep(wait_time)

    class Log(object):

        def __init__(self, file_name='log/%T', is_to_console=True):
            self.log_obj = None
            self.file_name = self.reformat_file_name(file_name)
            self.is_to_console = is_to_console

        @staticmethod
        def reformat_file_name(file_name):
            time_str = dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            if '%T' in file_name:
                file_name = file_name.replace('%T', time_str)
            if '%D' in file_name:
                file_name = file_name.replace('%D', time_str.split('T')[0])
            return file_name

        def start(self, is_print=False):
            self.log_obj = self.SavePrint(self.file_name, self.is_to_console)
            self.log_obj.start()
            if is_print:
                print '[log] log starts, to file %s' % (self.file_name,)

        def close(self):
            self.log_obj.close()

        def save(self, target, is_print=False):
            os.system('cp %s %s' % (self.file_name, target))
            if is_print:
                print '[log] log copy to %s' % (target,)

        class SavePrint(object):

            def __init__(self, files, is_to_console=True):

                self.is_to_console = is_to_console
                self.console = sys.__stdout__

                if isinstance(files, str):
                    files = [files]

                self.file_objects = [open(file_, 'w') for file_ in files]

            def write(self, message):
                for file_object in self.file_objects:
                    file_object.write(message)
                if self.is_to_console:
                    self.console.write(message)

            def flush(self):
                pass

            def start(self):
                sys.stdout = self

            def close(self):
                for file_object in self.file_objects:
                    file_object.close()
                sys.stdout = self.console

    class LogAnalysis(object):

        @staticmethod
        def single_re(log_str, re_expression, keys, functions=None):
            if isinstance(keys, str):
                keys = [keys]
            if functions is not None:
                assert len(keys) == len(functions)
            mappings = re.findall(re_expression, log_str)
            if not mappings:
                print 'Warning: no matches in log_str'
                return []

            elif len(mappings) >= 1:
                keys_dict_list = []
                for mapping in mappings:
                    if functions:
                        if len(keys) == 1:
                            keys_dict_list.append({keys[0]: functions[0](mapping)})
                        else:
                            assert len(keys) == len(mapping) == len(functions)
                            keys_dict_list.append({keys[i]: functions[i](mapping[i]) for i in range(len(keys))})
                    else:
                        if len(keys) == 1:
                            keys_dict_list.append({keys[0]: mapping})
                        else:
                            assert len(keys) == len(mapping)
                            keys_dict_list.append({keys[i]: mapping[i] for i in range(len(keys))})

                return keys_dict_list

    class Basic(object):

        @staticmethod
        def logWrapper(func, logFile, isToConsole=True):
            def funcWithLog(*args, **kwargs):
                logObject = RxModeling.Log(logFile, is_to_console=isToConsole)
                logObject.start()
                returnValue = func(*args, **kwargs)
                logObject.close()
                return returnValue

            return funcWithLog

        @staticmethod
        def timeWrapper(func):
            def funcWithTime(*args, **kwargs):
                timeObject = RxModeling.Time(is_margin=True)
                timeObject.show()
                returnValue = func(*args, **kwargs)
                timeObject.show()
                return returnValue

            return funcWithTime

        @staticmethod
        def getBound(breakPoints=(), lowBound=-np.inf, highBound=np.inf):
            bps = [lowBound] + list(breakPoints) + [highBound]
            return [(bps[i], bps[i + 1]) for i in range(len(breakPoints) + 1)]

        @staticmethod
        def floatToEvent(floatDf, breakPoints):
            markDf = pd.DataFrame(np.zeros_like(floatDf), index=floatDf.index, columns=floatDf.columns)
            breakBound = RxModeling.Basic.getBound(breakPoints, )
            for i in range(len(breakBound)):
                bound = breakBound[i]
                markDf[(floatDf > bound[0]) & (floatDf <= bound[1])] = i + 1
            return markDf.astype(int)

        @staticmethod
        def getValid(*arrays):
            valid = ~np.isnan(arrays[0])
            for a in arrays[1:]:
                valid = valid & (~np.isnan(a))
            return tuple(a[valid] for a in arrays)

        @staticmethod
        def iterUpdateDict(dictRaw, dictAdd):
            for k, v in dictAdd.iteritems():
                if k in dictRaw and isinstance(v, dict) and isinstance(dictRaw[k], dict):
                    dictRaw[k] = RxModeling.Basic.iterUpdateDict(dictRaw[k], dictAdd[k])
                else:
                    dictRaw[k] = v
            return dictRaw

    class VariableSelection(object):
        """
        cache keys:

            conflicts: dict
                for MarginF
                {'x1': ['x2', 'x3', ...]
                 'x4': ['x5']
                 'x10': ['x1]
                 ...
                }

            remove_x_path: list
                ['x1', 'x2', 'x3', ...]
        """

        class AbstractSelection(object):

            @staticmethod
            def _check_data(x, y):
                if isinstance(x, pd.DataFrame):
                    pass
                elif isinstance(x, np.ndarray):
                    x = pd.DataFrame(x)
                else:
                    raise TypeError('Unknown type of x')
                assert len(x.shape) == 2
                y = np.array(y).ravel()
                assert x.shape[0] == len(y)
                if len(y) < 100:
                    print 'Warning: data length %d too small ' % (len(y),)
                return x, y

            def select(self, x, y, x_columns=None, cache={}):
                x, y = self._check_data(x, y)
                if x_columns is None:
                    x_columns = list(x.columns)
                else:
                    x_columns = list(x_columns)
                    for x_column in x_columns:
                        assert x_column in x.columns
                return self._select(x, y, x_columns, cache=cache)

            def _select(self, x, y, x_columns, cache={}):
                raise NotImplementedError

        class RemoveAllConst(AbstractSelection):

            def __init__(self, is_print=False):
                self.is_print = is_print

            def _select(self, x, y, x_columns, cache={}):
                if self.is_print:
                    print '[Remove All Const] selecting ...'
                for x_column in x_columns:
                    x_single = x[x_column].values
                    if len(sm.add_constant(x_single).shape) == 1:
                        x_columns.remove(x_column)
                        if self.is_print:
                            print '[Remove All Const] %d remain, remove %s, all constant' \
                                  % (len(x_columns), x_column,)
                return x_columns

        class BackwardSingleP(AbstractSelection):

            def __init__(self, p_threshold=0.05, is_print=False):
                self.p_threshold = p_threshold
                self.is_print = is_print

            def _select(self, x, y, x_columns, cache={}):
                if self.is_print:
                    print '[Select Single P] selecting ...'
                for x_column in x_columns:
                    x_single = x[x_column].values
                    x_reg = sm.add_constant(x_single)
                    model = sm.OLS(y, x_reg).fit()
                    p_value = model.pvalues[-1]
                    if p_value > self.p_threshold:
                        x_columns.remove(x_column)
                        if self.is_print:
                            print '[Select Single P] %d remain, remove %s, single p value %.4f' \
                                  % (len(x_columns), x_column, p_value)
                return x_columns

        class BackwardMarginR2(AbstractSelection):
            def __init__(self, r2_diff_threshold=-np.infty, n_min=1, is_print=False):
                self.r2_diff_threshold = r2_diff_threshold
                self.n_min = n_min
                self.is_print = is_print

            def _select(self, x, y, x_columns, cache={}):
                if self.is_print:
                    print '[Select Margin R2] selecting ...'

                if len(x_columns) <= self.n_min:
                    return x_columns

                while len(x_columns) > self.n_min:

                    bench_r2 = sm.OLS(y, x[x_columns]).fit().rsquared_adj
                    best_r2_diff, best_x_column = -np.inf, None

                    for x_column in x_columns:
                        x_columns_tmp = x_columns[:]
                        x_columns_tmp.remove(x_column)
                        tmp_r2_diff = sm.OLS(y, x[x_columns_tmp]).fit().rsquared_adj - bench_r2
                        if tmp_r2_diff > best_r2_diff:
                            best_r2_diff, best_x_column = tmp_r2_diff, x_column

                    if best_r2_diff > self.r2_diff_threshold:
                        x_columns.remove(best_x_column)
                        if self.is_print:
                            print '[Select Margin R2] %d remain, remove %s, %.6f r2 diff' \
                                  % (len(x_columns), best_r2_diff, best_x_column)
                    else:
                        if self.is_print:
                            print '[Select Margin R2] %d remain, stops, %.6f r2 diff' \
                                  % (len(x_columns), best_r2_diff)
                        break
                return x_columns

        class BackwardMarginT(AbstractSelection):
            def __init__(self, t_threshold=np.infty, n_min=1, is_print=False):
                self.t_threshold = t_threshold
                self.n_min = n_min
                self.is_print = is_print

            def _select(self, x, y, x_columns, cache={}):
                if self.is_print:
                    print '[Select Margin T] selecting ... %d remain' % (len(x_columns),)
                    print '[Select Margin T] T threshold: %.4f, min num of var: %d' % (self.t_threshold, self.n_min)

                while len(x_columns) > self.n_min:

                    t_values = sm.OLS(y, x[x_columns]).fit().tvalues.abs().sort_values()
                    x_column, min_t_value = t_values.index[0], t_values[0]

                    if min_t_value < self.t_threshold:
                        x_columns.remove(x_column)
                        if self.is_print:
                            print '[Select Margin T] %d remain, remove %s, t value: %.4f' \
                                  % (len(x_columns), x_column, min_t_value)
                    else:
                        if self.is_print:
                            print '[Select Margin T] %d remain, stops, t value: %.4f' \
                                  % (len(x_columns), min_t_value)
                        break
                return x_columns

        class BackwardMarginF(AbstractSelection):

            def __init__(self, group_size=5, f_p_value=0.0, n_min=1, is_print=False):
                self.group_size = group_size
                self.f_p_value = f_p_value
                self.n_min = n_min
                self.is_print = is_print

            def _select(self, x, y, x_columns, cache={}):
                if self.is_print:
                    print '[Select Margin F] selecting ... %d remain' % (len(x_columns),)
                    print '[Select Margin F] group size: %d' % (self.group_size,)
                    print '[Select Margin F] F P-value: %.4f, min num of var: %d' % (self.f_p_value, self.n_min)
                while len(x_columns) > self.n_min:
                    bench = sm.OLS(y, x[x_columns]).fit()
                    p_values_sorted = bench.pvalues.sort_values(ascending=False)
                    for count_i in range(self.group_size):
                        x_name = p_values_sorted.index[count_i]
                        conflicts = cache.get('conflicts', {})
                        for x_other in conflicts.get(x_name, []):
                            try:
                                p_values_sorted.drop(x_other, inplace=True)
                            except:
                                pass
                    group = p_values_sorted[:self.group_size]
                    print '[Select Margin F]', list(group.index)
                    print '[Select Margin F]', list(group.values)
                    if self.f_p_value == 0.0:
                        for x_column in list(group.index):
                            x_columns.remove(x_column)
                            if self.is_print:
                                print '[Select Margin F] %d remain, remove %s, f p-value: %.4f' \
                                      % (len(x_columns), x_column, np.nan)
                    else:
                        restricted_model = sm.OLS(y, x[x_columns].drop(group.index, axis=1)).fit()
                        f_test_res = bench.compare_f_test(restricted_model)
                        print f_test_res
                        f_value = f_test_res[1]
                        if f_value > self.f_p_value:
                            for x_column in list(reversed(list(group.index))):
                                x_columns.remove(x_column)
                                if self.is_print:
                                    print '[Select Margin F] %d remain, remove %s, f p-value: %.4f' \
                                          % (len(x_columns), x_column, f_value)
                        else:
                            if self.is_print:
                                print '[Select Margin F] %d remain, stops, f value: %.4f' \
                                      % (len(x_columns), f_value)
                            break
                return x_columns

    class Fitting(object):

        @staticmethod
        def _testFit(fitObj=None, n=1000):
            if fitObj is None:
                fitObj = RxModeling.Fitting.PiecewiseRegContinous()
            x = np.random.rand(n) - 0.5
            epsilon = np.random.randn(n) * 0.1
            y = (x + 1) * (x > 0) + ((-1) * x + 1) * (x < 0) + epsilon
            x = np.array(list(x) + [np.nan])
            y = np.array(list(y) + [np.nan])
            fitObj.fit(x, y)
            yHat = fitObj.predict(x)
            plt.scatter(x, y)
            ss = np.argsort(x)
            plt.plot(x[ss], yHat[ss])
            plt.show()
            return {'fitObj': fitObj, 'yHat': yHat, 'x': x, 'y': y}

        class LinearRegression(LinearRegression):

            def fit(self, X, y, sample_weight=None):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                valid = np.all(~np.isnan(X), axis=1) & (~np.isnan(y))
                X, y, sample_weight = X[valid, :], y[valid], sample_weight[valid]
                super(RxModeling.Fitting.LinearRegression, self).fit(X, y, sample_weight)

            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                y = np.full(X.shape[0], np.nan)
                valid = np.all(~np.isnan(X), axis=1)
                X = X[valid, :]
                yValid = super(RxModeling.Fitting.LinearRegression, self).predict(X)
                y[valid] = yValid
                return y

        class DecisionTree(DecisionTreeRegressor):

            def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                valid = np.all(~np.isnan(X), axis=1) & (~np.isnan(y))
                X, y, sample_weight = X[valid, :], y[valid], sample_weight[valid]
                super(RxModeling.Fitting.DecisionTree, self).fit(X, y, sample_weight, check_input, X_idx_sorted)

            def predict(self, X, check_input=True):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                y = np.full(X.shape[0], np.nan)
                valid = np.all(~np.isnan(X), axis=1)
                X = X[valid, :]
                yValid = super(RxModeling.Fitting.DecisionTree, self).predict(X, check_input)
                y[valid] = yValid
                return y

        class RandomForest(RandomForestRegressor):

            def fit(self, X, y, sample_weight=None):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                valid = np.all(~np.isnan(X), axis=1) & (~np.isnan(y))
                X, y, sample_weight = X[valid, :], y[valid], sample_weight[valid]
                super(RxModeling.Fitting.RandomForest, self).fit(X, y, sample_weight=sample_weight)

            def predict(self, X):
                if X.ndim == 1:
                    X = X.reshape((-1, 1))
                y = np.full(X.shape[0], np.nan)
                valid = np.all(~np.isnan(X), axis=1)
                X = X[valid, :]
                yValid = super(RxModeling.Fitting.RandomForest, self).predict(X)
                y[valid] = yValid
                return y

        class PloyReg(object):

            def __init__(self, degree):
                self.degree = degree
                self.coef = None
                self.polyFunc = None

            def fit(self, xTrain, yTrain):
                xTrain, yTrain = RxModeling.Basic.getValid(xTrain, yTrain)
                polyArgs = np.polyfit(xTrain, yTrain, self.degree)
                self.coef = polyArgs
                self.polyFunc = lambda xReg: np.sum(
                    [self.coef[i] * xReg ** (self.degree - i) for i in range(self.degree + 1)])
                return self

            def predict(self, xTest):
                return np.array([self.polyFunc(x) for x in xTest])

        class PiecewiseRegContinous(object):

            def __init__(self, breakPoints=(), setBreakPointsInFit=False, setQuantileNum=5):
                self.breakPoints = breakPoints
                self.breakBound = RxModeling.Basic.getBound(breakPoints)

                self.setBreakPointsInFit = setBreakPointsInFit
                self.setQuantileNum = setQuantileNum
                self.model = None

            def setBreakByQuantile(self, xTrain, fracNum=5):
                xTrain = RxModeling.Basic.getValid(xTrain)
                self.breakPoints = [np.percentile(xTrain, int(i * 100. / fracNum)) for i in range(1, fracNum)]
                self.breakBound = RxModeling.Basic.getBound(self.breakPoints)

            def fit(self, xTrain, yTrain):
                xTrain, yTrain = RxModeling.Basic.getValid(xTrain, yTrain)
                if self.setBreakPointsInFit:
                    self.setBreakByQuantile(xTrain, self.setQuantileNum)
                dataDf = pd.DataFrame({'x': xTrain, 'y': yTrain})
                for i, bp in enumerate(self.breakPoints):
                    dataDf['x_' + str(i)] = np.where(xTrain > bp, xTrain - bp, 0)
                formula = 'y ~ ' + '+'.join(['x'] + ['x_' + str(i) for i in range(len(self.breakPoints))])
                model = smf.ols(formula=formula, data=dataDf).fit()
                self.model = model
                return self

            def predict(self, xTest):
                xTest = np.array(xTest)
                yHat = np.full(xTest.shape, np.nan)
                valid = ~np.isnan(xTest)
                xTest = xTest[valid]
                dataDf = pd.DataFrame({'x': xTest})
                for i, bp in enumerate(self.breakPoints):
                    dataDf['x_' + str(i)] = np.where(xTest > bp, xTest - bp, 0)
                yHat[valid] = self.model.predict(dataDf)
                return yHat

        class PiecewiseReg(object):

            def __init__(self, breakPoints=(), addConstant=True, setBreakPointsInFit=False, setQuantileNum=5):
                self.breakPoints = breakPoints
                self.breakBound = RxModeling.Basic.getBound(breakPoints)
                self.addConstant = addConstant

                self.setBreakPointsInFit = setBreakPointsInFit
                self.setQuantileNum = setQuantileNum

                self.models = None

            def setBreakByQuantile(self, xTrain, fracNum=5):
                xTrain = RxModeling.Basic.getValid(xTrain)
                self.breakPoints = [np.percentile(xTrain, int(i * 100. / fracNum)) for i in range(1, fracNum)]
                self.breakBound = RxModeling.Basic.getBound(self.breakPoints)

            def fit(self, xTrain, yTrain):
                xTrain, yTrain = RxModeling.Basic.getValid(xTrain, yTrain)
                data = pd.DataFrame({'x': xTrain, 'y': yTrain})
                datas = [data[(data['x'] > low) & (data['x'] < upper)] for low, upper in self.breakBound]
                self.models = [sm.OLS(d['y'].values,
                                      d['x'].values if not self.addConstant else sm.add_constant(d['x'].values)).fit()
                               for d in datas]
                return self

            def predict(self, xTest):
                xTest = np.array(xTest)
                yHat = np.full(xTest.shape, np.nan)
                for i, (low, upper) in enumerate(self.breakBound):
                    con = (xTest > low) & (xTest <= upper)
                    xCon = xTest[con]
                    if len(xCon) == 0:
                        continue
                    if self.addConstant:
                        xCon = sm.add_constant(xCon) if len(xCon) != 1 else np.array([1, xCon])
                    yHat[con] = self.models[i].predict(xCon if not self.addConstant else sm.add_constant(xCon))
                return yHat

        class marginRegression(object):

            def __init__(self, fit_intercept=True):
                self.fit_intercept = fit_intercept

            def fit(self, xTrain, yTrain, sample_weight=None):
                if xTrain.ndim == 1:
                    xTrain = xTrain.reshape((-1, 1))
                valid = np.all(~np.isnan(xTrain), axis=1) & (~np.isnan(yTrain))
                xTrain, yTrain, sample_weight = xTrain[valid, :], yTrain[valid], sample_weight[valid]
                coef = np.full(xTrain.shape[1], np.nan)
                intercept = 0.

                for i in range(xTrain.shape[1]):
                    xTmp = xTrain[:, i]
                    lr = RxModeling.Fitting.LinearRegression(fit_intercept=self.fit_intercept)
                    lr.fit(xTmp, yTrain, sample_weight)
                    if self.fit_intercept:
                        intercept += lr.coef_[0]
                        coef[i] = lr.coef_[1]
                    else:
                        pass

            def predict(self, xTest):
                pass

        @staticmethod
        def normalizeByVectors(rawDf, vectorDfs, addConstant=True, minObs=100):
            vectors = [vdf.loc[rawDf.index, rawDf.columns].values for vdf in vectorDfs]
            if addConstant:
                vectors.insert(0, np.ones(rawDf.shape))

            vectorPanel = pd.Panel(vectors, major_axis=rawDf.index, minor_axis=rawDf.columns)

            coefDict = {}
            resDf = pd.DataFrame(index=rawDf.index, columns=rawDf.columns)
            for idx, rawLine in rawDf.iterrows():
                y = rawLine.values
                x = vectorPanel.loc[:, idx, :].values
                valid = np.all(~np.isnan(x), axis=1) & (~np.isnan(y))
                if np.sum(valid) <= minObs:
                    continue
                xReg, yReg = x[valid], y[valid]
                regModel = sm.OLS(yReg, xReg).fit()
                coefDict[idx] = regModel.params
                res = np.full(y.shape, np.nan, )
                res[valid] = regModel.resid
                resDf.loc[idx] = res
            return {'residual': resDf, 'coefDict': coefDict}

        @staticmethod
        def normalizeByVectorPoly(rawDf, vectorDf, degree=3, minObs=100):
            coefDict = {}
            resDf = pd.DataFrame(index=rawDf.index, columns=rawDf.columns)
            for idx, rawLine in rawDf.iterrows():
                y = rawLine.values
                x = vectorDf.loc[idx].values
                valid = (~np.isnan(x)) & (~np.isnan(y))
                if np.sum(valid) <= minObs:
                    continue
                xValid, yValid = x[valid], y[valid]
                polyArgs = np.polyfit(xValid, yValid, degree)
                coefDict[idx] = polyArgs
                polyFunc = lambda x: np.sum([polyArgs[i] * x ** (degree - i) for i in range(degree + 1)])

                res = np.full(y.shape, np.nan, )
                res[valid] = yValid - np.array([polyFunc(xi) for xi in xValid])
                resDf.loc[idx] = res
            return {'residual': resDf, 'coefDict': coefDict}

    class X(object):

        @staticmethod
        def calc_outr2(y, y_hat):
            y, y_hat = np.array(y), np.array(y_hat)
            return 1 - np.nansum((y - y_hat) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)

        @staticmethod
        def calc_basic_statistics(x, info=None):
            info = ['mean', 'std', 'skew', 'kurt', 'num', 'nanNum', 'max', 'min', 'num_out2std', 'num_out3std',
                    'num_out5std', 'num_out10std'] \
                if info is None else info

            x = np.array(x).ravel()
            nanNum = np.sum(np.isnan(x))
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return pd.Series({i: np.nan for i in info}, index=info)

            func_map = {'mean': np.mean,
                        'std': np.std,
                        'skew': stats.skew,
                        'kurt': stats.kurtosis,
                        'num': len,
                        'nanNum': lambda _: nanNum,
                        'max': np.max,
                        'min': np.min,
                        'num_out2std': lambda x_func: np.sum((x_func - np.mean(x_func)) > 2 * np.std(x_func)) +
                                                      np.sum((x_func - np.mean(x_func)) < -2 * np.std(x_func)),
                        'num_out3std': lambda x_func: np.sum((x_func - np.mean(x_func)) > 3 * np.std(x_func)) +
                                                      np.sum((x_func - np.mean(x_func)) < -3 * np.std(x_func)),
                        'num_out5std': lambda x_func: np.sum((x_func - np.mean(x_func)) > 5 * np.std(x_func)) +
                                                      np.sum((x_func - np.mean(x_func)) < -5 * np.std(x_func)),
                        'num_out10std': lambda x_func: np.sum((x_func - np.mean(x_func)) > 10 * np.std(x_func)) +
                                                       np.sum((x_func - np.mean(x_func)) < -10 * np.std(x_func)),
                        }
            basic_statistic_dict = {key: func_map[key](x) for key in info if key in func_map}
            basic_statistic_series = pd.Series(basic_statistic_dict, index=info)
            return basic_statistic_series

        @staticmethod
        def getConDfs(dfDict, condition, fillNaValue=-1):
            if isinstance(condition, np.ndarray):
                tmpdf = dfDict.values()[0]
                condition = pd.DataFrame(condition, tmpdf.index, tmpdf.columns)
            return pd.DataFrame({k: dfDict[k].fillna(fillNaValue)[condition == True].stack() for k in dfDict})

    class XY(object):

        @staticmethod
        def getEventDecay(xEvent, yPanel, validDf=None, includeZero=False):
            validDf = yPanel[1].notnull() if validDf is None else validDf
            days = yPanel.items
            events = pd.melt(xEvent).value.dropna().unique()
            events = sorted(events if includeZero else events[events != 0])
            eventDecay = pd.DataFrame(index=events, columns=days)
            spotNum = pd.Series(index=events)
            for event in events:
                print event,
                isEvent = pd.DataFrame(xEvent == event) & validDf
                spotNum[event] = isEvent.sum().sum()
                for fday in days:
                    print fday,
                    tmpReturn = yPanel[fday].loc[xEvent.index, xEvent.columns]
                    tmpReturn[~isEvent] = np.nan
                    eventDecay.loc[event, fday] = tmpReturn.fillna(0).sum().sum() / spotNum[event]
                print
            return eventDecay, spotNum

        @staticmethod
        def plotEventDecay(eventDecay, spotNum, eventNameMap={}, **plotkwargs):
            events = list(eventDecay.index)
            maxY, minY = 0., 0.
            for event in events:
                label = '%s, num%d' % (eventNameMap[event], spotNum[event]) if event in eventNameMap else \
                    'event%d, num%d' % (event, spotNum[event])
                eventDecay.loc[event].plot(label=label, **plotkwargs)
                maxY = max(maxY, eventDecay.loc[event].max())
                minY = min(minY, eventDecay.loc[event].min())
            ax = plt.gca()
            ax.legend(loc='best')
            ax.plot([0, 0], [minY, maxY], )
            ax.set_title('decay')

        @staticmethod
        def corrCurve(yDict, xSeries, lookBackLength):
            dataIdx = xSeries.index
            xDf = pd.DataFrame({i: xSeries.shift(i) for i in range(lookBackLength)}, index=dataIdx)
            yDf = pd.DataFrame(yDict)
            corrDf = pd.DataFrame(index=range(lookBackLength), columns=yDf.columns)
            for y in yDf:
                corrDf[y] = xDf.corrwith(yDf[y])

            corrSum = corrDf.cumsum()
            corrSum.plot()
            return {'xDf': xDf, 'yDf': yDf, 'corrDf': corrDf}

    class NpTools(object):

        @staticmethod
        def rankNan(x):
            y = np.argsort(np.where(np.isnan(x), np.inf, x), axis=0)
            y2 = np.full(x.shape, np.nan, )
            rankArray = np.arange(1, y.shape[0] + 1)
            for i, j in enumerate(y.T):
                y2[:, i][j] = rankArray
            y2[np.isnan(x)] = np.nan
            return y2

        @staticmethod
        def getPolyFunc(polyArgs):
            degree = len(polyArgs) - 1
            polyFunc = lambda x: np.sum([polyArgs[i] * x ** (degree - i) for i in range(degree + 1)])
            return polyFunc

        @staticmethod
        def divide_into_group(arr, group_num=None, group_size=None):
            if group_num is not None:
                group_num = int(group_num)
                assert group_size is None
                group_size_small = len(arr) / group_num
                group_num_big = (len(arr) % group_num)
                nums = [(group_size_small + 1 if i < group_num_big else group_size_small)
                        for i in range(group_num)]
                nums.insert(0, 0)
            elif group_size is not None:
                group_size = int(group_size)
                group_num = int(np.ceil(len(arr) * 1.0 / group_size))
                nums = [group_size] * (len(arr) / group_size) + [(len(arr) % group_size)]
                nums.insert(0, 0)
            else:
                raise Exception
            indexs = np.cumsum(np.array(nums))
            new_arr = []
            for i in range(group_num):
                new_arr.append(arr[indexs[i]:indexs[i + 1]])
            return new_arr

        @staticmethod
        def checkSame(matrix1, matrix2, maxDiff=1e-8, isNan=True, isPrint=True, ):
            matrix1, matrix2 = np.array(matrix1, dtype=float), np.array(matrix2, dtype=float)
            assert matrix1.shape == matrix2.shape
            res = {}
            if isNan:
                nan1 = np.isnan(matrix1) & (~np.isnan(matrix2))
                nan2 = (~np.isnan(matrix1)) & np.isnan(matrix2)
                res['nan1'] = nan1
                res['nan2'] = nan2
                if isPrint:
                    print 'matrix1 nan alone:', np.sum(nan1)
                    print 'matrix2 nan alone:', np.sum(nan2)
            diff = (np.abs(matrix1 - matrix2) >= maxDiff)
            res['diff'] = diff
            if isPrint:
                print 'different values:', np.sum(diff)
            return res

        @staticmethod
        def countChangePoints(series, isPrint=True):
            """
            :return:{'changeNum': len(changePoints),
                    'changePoints': changePoints}
            """
            array = np.array(series)
            changePoints = []
            lastState = np.isnan(array[0])
            for i in range(1, len(array)):
                newState = np.isnan(array[i])
                if newState ^ lastState:
                    changePoints.append(i)
                    lastState = newState
            if isPrint:
                print 'change points:', len(changePoints)
            return {'changeNum': len(changePoints),
                    'changePoints': changePoints}

    class PdTools(object):

        @staticmethod
        def getZscore(df, axis=1, level=2):
            if level == 1:
                return df.sub(df.mean(axis=axis), axis=1 - axis).div(df.abs().sum(axis=axis), axis=1 - axis)

            if level == 2:
                return df.sub(df.mean(axis=axis), axis=1 - axis).div(df.std(axis=axis), axis=1 - axis)

        @staticmethod
        def maskNotEnough(df, num, axis=0, ):
            dfReturn = df.copy()
            notValid = df.notnull().sum(axis=axis) < num
            if axis == 0:
                dfReturn.loc[:, notValid] = np.nan
            elif axis == 1:
                dfReturn[notValid] = np.nan
            return dfReturn

        @staticmethod
        def ffillDecay(df, decayRate=1., limit=None):
            dfFillNa = df.ffill(limit=limit)
            if np.isclose(decayRate, 1., 1.e-8, 1.e-8, ):
                return dfFillNa
            else:
                dfNanSum = df.isnull().astype(int).cumsum()
                dfNanLength = dfNanSum - dfNanSum[df.notnull()].ffill().fillna(0)
                ratio = decayRate ** dfNanLength
                return dfFillNa.mul(ratio)

        @staticmethod
        def ffillDecayMulti(df, decayRate=1., limit=None):
            if np.isclose(decayRate, 1., 1.e-8, 1.e-8, ):
                return df.ffill(limit=limit)
            else:
                isNa = df.ffill(limit=limit).isnull()
                dfFill0 = df.fillna(0.)
                dfFill0ewma = dfFill0.ewm(alpha=1 - decayRate, adjust=False).mean()
                dfFill0ewma[isNa] = np.nan
                return dfFill0ewma

        @staticmethod
        def countNan(df):
            dfNanSum = df.isnull().astype(int).cumsum()
            dfNanLength = dfNanSum - dfNanSum[df.notnull()].ffill().fillna(0)
            return dfNanLength

        @staticmethod
        def qcut(df, qNum, labels=None, returnBins=False):
            labels = range(1, qNum + 1) if labels is None else labels
            qcutDf = pd.DataFrame(np.nan, df.index, df.columns)
            if returnBins:
                binsDf = pd.DataFrame(np.nan, df.index, range(qNum + 1))
            for idx, line in df.iterrows():
                lineNa = line.dropna()
                if len(lineNa) == 0:
                    continue
                try:
                    res = pd.qcut(lineNa, qNum, labels, returnBins, )
                    if returnBins:
                        qcutDf.loc[idx][res[0].index] = res[0]
                        binsDf.loc[idx] = res[1]
                    else:
                        qcutDf.loc[idx][res.index] = res
                except:
                    columnsGroup = RxModeling.NpTools.divide_into_group(list(lineNa.sort_values().index), qNum)
                    res = pd.Series(index=lineNa.index)
                    for label, gcolumns in zip(labels, columnsGroup):
                        res[gcolumns] = label
                    qcutDf.loc[idx][res.index] = res
                    if returnBins:
                        binsDf.loc[idx] = [lineNa[gcolumns[0]] for gcolumns in columnsGroup] + [
                            lineNa[columnsGroup[-1][-1]]]

            return {'qcutDf': qcutDf, 'binsDf': binsDf} if returnBins else qcutDf

        @staticmethod
        def showNear(dfs, dfNames, recordIndex, backNum=0, forwardNum=0, column=None, ):
            recordIdx = dfs[0].index.get_loc(recordIndex)
            startIdx = max(0, recordIdx - backNum)
            endIdx = min(len(dfs[0]), recordIdx + forwardNum)
            return pd.DataFrame({dfName: df[column].iloc[startIdx:endIdx] for df, dfName in zip(dfs, dfNames)})

        @staticmethod
        def winsorize(df, limits=(0.05, 0.05), winType='percent', axis=1):
            """
            winType: 'percent'('p') or 'std'('s')
            """
            if winType in ('std', 's'):
                mean = df.mean(axis=axis)
                std = df.std(axis=axis)
                lowerLimit = (mean - limits[0] * std).fillna(-np.inf)
                upperLimit = (mean + limits[1] * std).fillna(np.inf)
            elif winType in ('percent', 'p'):
                dfQuantile = pd.DataFrame(index=[limits[0], 1 - limits[1]], columns=df.index)
                for idx in df.index:
                    dfQuantile[idx] = df.loc[idx].dropna().quantile([limits[0], 1 - limits[1]])
                lowerLimit = dfQuantile.iloc[0].fillna(-np.inf)
                upperLimit = dfQuantile.iloc[1].fillna(np.inf)
            else:
                raise Exception('Unknown winType %s' % (winType,))
            return df.clip(lowerLimit, upperLimit, axis=1 - axis)

        @staticmethod
        def getNearPanel(df, backNum=1, forwardNum=1):
            idxNum = range(-backNum, forwardNum + 1)
            return pd.Panel({idx: df.shift(-idx) for idx in idxNum})

        @staticmethod
        def getValueByGroup(groupValue, groupDf, ):
            """
            :param groupValue: dates x groups, values: value
            :param groupDf: dates x symbols, values: group
            :return:
            """
            recordsGroup = groupDf.stack().reset_index()
            recordsGroup.columns = ['dates', 'symbols', 'group']

            recordsValue = groupValue.stack().reset_index()
            recordsValue.columns = ['dates', 'group', 'value']

            recordsAll = pd.merge(recordsGroup, recordsValue, how='left', on=['dates', 'group'])
            return recordsAll.pivot('dates', 'symbols', 'value').loc[groupDf.index, groupDf.columns]

        @staticmethod
        def calGroupInfo(dataDf, groupDf, funcs):
            if isinstance(groupDf, pd.Series):
                groupDf = pd.DataFrame(np.repeat(groupDf.loc[dataDf.columns].values.reshape(1, -1),
                                                 len(dataDf.index), axis=0),
                                       index=dataDf.index, columns=dataDf.columns)
            groups = sorted(list(groupDf.stack().unique()))
            result = pd.Panel(items=funcs.keys(), major_axis=dataDf.index, minor_axis=groups)
            for group in groups:
                dataDfGroup = dataDf[groupDf == group]
                for funcName in funcs:
                    result.loc[funcName, :, group] = funcs[funcName](dataDfGroup)
            return result

    class Time(object):
        def __init__(self, is_now=False, is_all=False, is_margin=False):
            self.start_time = None
            self.last_time = None
            self.is_now = is_now
            self.is_all = is_all
            self.is_margin = is_margin

        def show(self):
            now = dt.datetime.now()
            if self.start_time is None:
                self.start_time = now
                print '[Time] Start at:', now
                if self.is_margin:
                    self.last_time = now
            else:
                if self.is_now:
                    print '[Time] now:', now
                if self.is_all:
                    print '[Time] Since start:', now - self.start_time
                if self.is_margin:
                    print '[Time] Since last call:', now - self.last_time
                    self.last_time = now

    class Plot(object):

        @staticmethod
        def pie(series, names=None, num=None, is_sorted=True, figKwargs=None, pieKwargs=None):
            """
            :param series: pandas.series
            :param names: None, list, func
            if None:
                series.index
            if func:
                func(i) for i in series.index
            :param num: None or int
            """
            if callable(names):
                names = [names(i) for i in series.index]
            elif names is None:
                names = series.index

            series = series.copy()
            series.index = names

            if num is not None:
                series = series.sort_values(ascending=False)
                if num < len(series):
                    othersNum = np.sum(series[num - 1:])
                    series = series[:num - 1]
                    series['OTHERS'] = othersNum

            if is_sorted:
                series.sort_values(ascending=False, inplace=True)

            plt.figure(**({} if figKwargs is None else figKwargs))
            plt.pie(series.values, labels=series.index, **({} if pieKwargs is None else pieKwargs))

        @staticmethod
        def plotQuantile(x, y, plotNum=20, isReg=True, isStd=False, isShowCorr=False, **plotKwargs):

            x, y = np.array(x).ravel(), np.array(y).ravel()
            valid = (~np.isnan(x)) & (~np.isnan(y))
            x, y = x[valid], y[valid]
            xArg = np.argsort(x)
            x, y = x[xArg], y[xArg]
            xMean = np.array([np.mean(x[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
            yMean = np.array([np.mean(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
            df = pd.DataFrame({'x': xMean, 'y': yMean})
            df.plot.scatter('x', 'y', **plotKwargs)
            plt.title('quantile plot')

            if isStd:
                yStd = np.array(
                    [np.std(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)], ddof=1) for i in range(plotNum)])
                plt.fill_between(xMean, yMean + yStd, yMean - yStd, alpha=0.3)

            if isReg:
                model = sm.OLS(yMean, sm.add_constant(xMean)).fit()
                yHat = xMean * model.params[1] + model.params[0]
                plt.plot(xMean, yHat)

            if isShowCorr:
                ax = plt.gca()
                ax.text(0.01, 0.99, 'corr: %s' % (np.corrcoef(x, y)[0, 1],),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes, color='red', size=16)

            return df

        @staticmethod
        def polyfit(x, y, degree, plotNum=100, **plotKwargs):
            x, y = RxModeling.NpTools.getValid((x, y))
            xArg = np.argsort(x)
            x, y = x[xArg], y[xArg]
            polyArgs = np.polyfit(x, y, deg=degree)
            polyFunc = RxModeling.NpTools.getPolyFunc(polyArgs)
            xMean = np.array([np.mean(x[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
            yMean = np.array([np.mean(y[i * (len(x) / plotNum):(i + 1) * (len(x) / plotNum)]) for i in range(plotNum)])
            df = pd.DataFrame({'x': xMean, 'y': yMean})
            df.plot.scatter('x', 'y', **plotKwargs)
            plt.title('quantile plot')
            x = list()

            xMin, xMax = np.nanmin(xMean), np.nanmax(xMean)
            xList = np.arange(xMin, xMax, step=(xMax - xMin) * 1. / 300)
            yHatList = [polyFunc(x) for x in xList]
            plt.plot(xList, yHatList)
            return {'polyArgs': polyArgs,
                    'polyFunc': polyFunc,
                    'quantileDf': df,}

        @staticmethod
        def plot3D(df, **plotKwargs):
            x = range(df.shape[0])
            y = range(df.shape[1])
            z = df.values.T
            x1, y1 = np.meshgrid(x, y)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            from matplotlib import cm
            surf = ax.plot_surface(x1, y1, z, rstride=1, cstride=1, cmap=cm.coolwarm, antialiased=False,
                                   **plotKwargs)
            plt.xticks(x, df.index)
            plt.yticks(y, df.columns)
            fig.colorbar(surf, shrink=0.5, aspect=5)

        @staticmethod
        def plotBox(dataDf, axis=1, **figKwargs):
            if axis == 0:
                dataDf = dataDf.T
            dataList = [line.values.ravel() for num, line in dataDf.iterrows()]
            dataList2 = [d[~np.isnan(d)] for d in dataList]
            plt.figure(**figKwargs)
            sns.boxplot(data=dataList2, )

        @staticmethod
        def plotFunc(func, start=0, end=1, step=0.01):
            x = np.arange(start, end, step)
            y = np.array([func(i) for i in x])
            plt.plot(x, y)

    class Test(object):

        @staticmethod
        def pair_test(series1, series2, series1_name='series1', series2_name='series2',
                      level=0.05, is_plot=True, is_print=True):
            assert len(series1) == len(series2)
            if len(series1) <= 100:
                print 'Warning: length of data is %d, smaller than 100' % (len(series1),)
            dif = np.array(series1) - np.array(series2)
            dif_cum = np.cumsum(dif)
            corr1 = np.corrcoef(series1, series2)[0, 1]
            t_value = np.float(np.mean(dif) / np.sqrt(np.var(dif) / len(dif)))
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), len(dif)))
            if is_plot:
                fig = plt.figure(figsize=(20, 15))
                fig.suptitle('Pair Test')
                ax = fig.add_subplot(211)
                plt.plot(np.cumsum(series1), 'b')
                plt.plot(np.cumsum(series2), 'g')
                plt.title('Cum Return')
                plt.legend([series1_name, series2_name], loc='best')
                ax.text(0.01, 0.99, 'data length: %d' % (len(series1)),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes, color='red', size=16)
                ax = fig.add_subplot(212)
                plt.plot(dif_cum)
                plt.title('Diff Cum Return')
                ax.text(0.01, 0.99, 't_value: %0.4f\np_value: %0.4f\ncorr: %0.4f' % (t_value, p_value, corr1),
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes, color='red', size=16)
                plt.show()
            cv = stats.norm.ppf(1 - level / 2)
            is_h0_true = False if p_value < level else True
            if is_print:
                print ''
                print '*******  Pair T TEST  *******'
                if is_h0_true:
                    print 'h0 is True: diff is not significant'
                else:
                    print 'h0 is False: diff is significant'
                print 'p value: %f' % (p_value,)
                print 't stat: %f' % (t_value,)
                print 'critical value: %f' % (cv,)
            return is_h0_true, p_value, t_value, cv

        @staticmethod
        def jb_test(series, level=0.05, is_print=True):
            """
            output: (is_h0_true, p_value, jb_stat, critical value)
            """
            series = series[~np.isnan(series)]
            if len(series) < 100:
                print 'Warning(in JB test): data length: %d' % (len(series),)
            skew = stats.skew(series)
            kurt = stats.kurtosis(series)
            n = len(series)
            jb = (n - 1) * (skew ** 2 + kurt ** 2 / 4) / 6
            p_value = 1 - stats.chi2.cdf(jb, 2)
            cv = stats.chi2.ppf(1 - level, 2)
            is_h0_true = False if p_value < level else True
            if is_print:
                print ''
                print '*******  JB TEST  *******'
                print 'skew: %.4f' % (skew,)
                print 'kurt: %.4f' % (kurt,)
                if is_h0_true:
                    print 'h0 is True: data is normal'
                else:
                    print 'h0 is False: data is not normal'
                print 'p value: %f' % (p_value,)
                print 'jb stat: %f' % (jb,)
                print 'critical value: %f' % (cv,)
            return is_h0_true, p_value, jb, cv

        @staticmethod
        def box_test(series, lag=10, type_='ljungbox',
                     level=0.05, is_plot=True, is_print=True):
            """
            output: (is_h0_true, p_value, q_stat, critical value)
            """
            series = series[~np.isnan(series)]
            acf = sm.tsa.acf(series, nlags=lag)
            if is_plot:
                sm.graphics.tsa.plot_acf(series, lags=lag)
                plt.show()
            q_stat = sm.tsa.q_stat(acf[1:], len(series), type=type_)[0][-1]
            p_value = stats.chi2.sf(q_stat, lag)
            cv = stats.chi2.ppf(1 - level, lag)
            is_h0_true = False if p_value < level else True
            if is_print:
                print ''
                print '*******  Ljung Box TEST  *******'
                if is_h0_true:
                    print 'h0 is True: data is independent'
                else:
                    print 'h0 is False: data is not independent'
                print 'p value: %f' % (p_value,)
                print 'q stat: %f' % (q_stat,)
                print 'critical value: %f' % (cv,)
            return is_h0_true, p_value, q_stat, cv

    class StatisticTools(object):
        """
        normality test: JB test
        auto-correlation test: Box test
        """

        @staticmethod
        def find_pca_order(x, thresholds=None, is_plot=True):
            """
            input:
                thresholds: must has attr '__len__'
                    default [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
            """
            if thresholds is None:
                thresholds = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, ]
            assert hasattr(thresholds, '__len__')

            pca = PCA()
            pca.fit(x)
            ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

            print '-' * 50
            i, j = 0, 0
            nums = []
            while i < len(thresholds) and j < len(ratio_cumsum):

                if ratio_cumsum[j] < thresholds[i]:
                    j += 1
                else:
                    print 'thres:', thresholds[i], '\t\tnums:', j
                    i += 1
                    nums.append(j)

            print '-' * 50

            if is_plot:
                plt.plot(pca.explained_variance_ratio_, label='ratio')
                plt.plot(ratio_cumsum, label='ratio_cumsum')
                plt.legend(loc='best')
                plt.show()

            return pca

        @staticmethod
        def find_lasso_para(x, y, paras=None, start_exp=-10, end_exp=-1, ):
            """
            Output:
                test_paras, variable_num, coefs
            """
            x = np.array(x)
            y = np.array(y)
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            if paras is None:
                assert isinstance(start_exp, int)
                assert isinstance(end_exp, int)
                assert end_exp >= start_exp
                paras = [10 ** i for i in range(start_exp, end_exp)]
            variable_num = []
            params = []
            for para in paras:
                tmp_model = LassoLars(alpha=para)
                tmp_model.fit(sm.add_constant(x), y)
                tmp_coef = tmp_model.coef_
                variable_num.append(np.sum(tmp_coef != 0))
                params.append(tmp_coef)
            return paras, variable_num, params
