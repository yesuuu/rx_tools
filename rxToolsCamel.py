import sys
import time
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy.stats as stat
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LassoLars, LinearRegression
from sklearn.decomposition import PCA

matplotlib.use("Qt4Agg")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 15)


class RxToolsBasic(object):
    class Wrappers(object):

        @staticmethod
        def save_fig_wrapper(func):
            """
            A decorator to save pictures
            in func, you should plot a single figure, neither show nor save_fig.

            Notes
            -------------------------------------
            in the function, 'fig_save_file' should not be kwargs

            'fig_save_file' means picture path
            """

            def new_func(*args, **kwargs):

                if 'fig_save_file' in kwargs:
                    save_file = kwargs['fig_save_file']
                    del kwargs['fig_save_file']
                else:
                    save_file = None
                func(*args, **kwargs)
                if len(plt.get_fignums()) != 1:
                    raise Exception('plt model has more than one picture')
                if not save_file:
                    plt.show()
                else:
                    save_path = os.path.split(save_file)[0]
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    plt.savefig(save_file)
                    plt.close()

            return new_func

        @staticmethod
        def calc_time_wrapper(func):

            def new_func(*args, **kwargs):
                time1 = time.clock()
                result = func(*args, **kwargs)
                time2 = time.clock()
                diff_time = time2 - time1
                print 'exec time: %.8f s' % (diff_time,)
                return result

            return new_func

    class NpTools(object):

        @staticmethod
        def divide_into_group(arr, group_num=None, group_size=None):
            assert group_size is not None or group_num is not None
            if group_num is not None:
                group_num = int(group_num)
                assert group_size is None
                group_size_small = len(arr) / group_num
                group_num_big = (len(arr) % group_num)
                nums = [(group_size_small + 1 if i < group_num_big else group_size_small)
                        for i in range(group_num)]
                nums.insert(0, 0)
            if group_size is not None:
                group_size = int(group_size)
                assert group_num is None
                group_num = int(np.ceil(len(arr) * 1.0 / group_size))
                nums = [group_size] * (len(arr) / group_size) + [(len(arr) % group_size)]
                nums.insert(0, 0)
            indexs = np.cumsum(nums)
            new_arr = []
            for i in range(group_num):
                new_arr.append(arr[indexs[i]:indexs[i + 1]])
            return new_arr

        @staticmethod
        def get_quantile(x, y=None, quantile=0.5):
            y = x if y is None else y
            assert len(x) == len(y)
            x_arg = np.argsort(x)
            idx = int(round(len(x) * quantile))
            idx = idx - 1 if idx > 0 else 0
            return y[x_arg[idx]]

    class DevelopTools(object):

        @staticmethod
        def interrupt(message):
            stop_message = '\nDo you want to stop?(y or n):'
            response = raw_input(message + stop_message)
            if response in ('YES', 'Y', 'Yes', 'yes', 'y'):
                raise Exception('Stop!')
            else:
                return

        class PrintLogObject(object):
            def __init__(self, files, is_to_console=True):

                self.is_to_console = is_to_console
                self.console = sys.__stdout__

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

            def reset(self):
                for file_object in self.file_objects:
                    file_object.close()
                sys.stdout = self.console


class RxTools(RxToolsBasic):
    """
    Some useful tools from FRX.
    """

    class YHatEvaluate(object):

        @staticmethod
        def calcOutR2(y, yHat):
            y = np.array(y).ravel()
            yHat = np.array(yHat).ravel()
            return 1 - np.sum((np.array(y) - yHat) ** 2) / np.sum((y - np.mean(y)) ** 2)

        @staticmethod
        def calcTopMean(y, yHat, topPercentage=0.05, topType='top'):
            y, yHat = np.array(y).ravel(), np.array(yHat).ravel()
            if topType in ('top', 't', 'TOP', 'T', 'Top'):
                args = np.argsort(yHat)[-int(round(len(y) * topPercentage)):]
            elif topType in ('bottom', 'b', 'BOTTOM', 'B', 'Bottom'):
                args = np.argsort(yHat)[:int(round(len(y) * topPercentage))]
            else:
                raise Exception('Unknown top_type!')
            return np.mean(y[args])

        @staticmethod
        def plotQuantile(y, yHat, num=20, isReg=False):
            xArg = np.argsort(yHat)
            xSorted, ySorted = yHat[xArg], y[xArg]
            xMean = np.array(
                [np.mean(xSorted[i * (len(yHat) / num):(i + 1) * (len(yHat) / num)]) for i in range(num)])
            yMean = np.array(
                [np.mean(ySorted[i * (len(yHat) / num):(i + 1) * (len(yHat) / num)]) for i in range(num)])
            plt.scatter(xMean, yMean)
            plt.title('qq plot')
            if isReg:
                model = LinearRegression()
                model.fit(xMean.reshape((-1, 1)), yMean)
                yHat = model.predict(xMean.reshape((-1, 1))).ravel()
                plt.plot(xMean, yHat)

        @staticmethod
        def printSummary(y, yHat,
                         isR2=True,
                         isTopMean=True, topList=(0.02, 0.01, 0.005, 0.0025),
                         isPlotQuantile=False, plotNum=20, plotIsReg=True):
            if isR2:
                print 'R2: %.8f' % (np.float(RxTools.YHatEvaluate.calcOutR2(y, yHat)), )
            if isTopMean:
                for topPercentage in topList:
                    topMean = (RxTools.YHatEvaluate.calcTopMean(y, yHat, topPercentage, 'top') -
                               RxTools.YHatEvaluate.calcTopMean(y, yHat, topPercentage, 'bottom')) / 2.
                    print '%-*f %f' % (10, topPercentage, topMean)
            if isPlotQuantile:
                RxTools.YHatEvaluate.plotQuantile(y, yHat, num=plotNum, isReg=plotIsReg)
                plt.show()

    class VariableSelection(object):

        class AbstractSelection(object):
            @staticmethod
            def _checkData(x, y):
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

            def select(self, x, y, xColumns=None, isPrint=False):
                x, y = self._checkData(x, y)
                if xColumns is None:
                    xColumns = list(x.columns)
                else:
                    xColumns = list(xColumns)
                    for xColumn in xColumns:
                        assert xColumn in x.columns
                return self._select(x, y, xColumns, isPrint)

            @abstractmethod
            def _select(self, x, y, xColumns, isPrint=False):
                raise NotImplementedError

        class RemoveAllConst(AbstractSelection):
            def _select(self, x, y, xColumns, isPrint=False):
                for xColumn in xColumns:
                    xSingle = x[xColumn].values
                    if len(sm.add_constant(xSingle).shape) == 1:
                        xColumns.remove(xColumn)
                        if isPrint:
                            print '[Remove All Const] %s dropped' % (xColumn,)
                return xColumns

        class BackwardSinglePValue(AbstractSelection):
            def __init__(self, pThreshold=0.05, isAddConstant=True):
                self.pThreshold = pThreshold
                self.isAddConstant = isAddConstant

            def _select(self, x, y, xColumns, isPrint=False):
                for x_column in xColumns:
                    x_single = x[x_column].value
                    x_reg = sm.add_constant(x_single) if self.isAddConstant else x_single
                    model = sm.OLS(y, x_reg).fit()
                    p_value = model.pvalues[-1]
                    if p_value > self.pThreshold:
                        xColumns.remove(x_column)
                        if isPrint:
                            print '[Remove All Const] %s dropped, single p value %.4f' % (x_column, p_value)
                return xColumns

        class BackwardMarginTValue(AbstractSelection):
            def __init__(self, t_threshold=1.):
                self.t_threshold = t_threshold

            def _select(self, x, y, x_columns, is_print=False):
                for x_column in x_columns:
                    x_single = x[x_column].value
                    x_reg = sm.add_constant(x_single) if self.add_constant else x_single
                    model = sm.OLS(y, x_reg).fit()
                    p_value = model.pvalues[-1]
                    if p_value > self.p_threshold:
                        x_columns.remove(x_column)
                        if is_print:
                            print '[Remove All Const] %s dropped, single p value %.4f' % (x_column, p_value)
                return x_columns


    class DataPrepare(object):
        def __init__(self, data, xColumns=None, yColumn='y',
                     trainRange=(0.4, 1.), validRange=(0.2, 0.4), testRange=(0., 0.2),
                     isNormalize=False, trainSampleGap=1, isInitNow=False, ):

            # init data and data_file
            if isinstance(data, str):
                self.dataFile = data
                self.data = None
            else:
                self.dataFile = None
                if isinstance(data, np.ndarray) or isinstance(data, list):
                    self.data = pd.DataFrame(data)
                elif isinstance(data, pd.DataFrame):
                    self.data = data
                else:
                    raise Exception('Unknown type of data!')

            self.trainSampleGap = trainSampleGap
            self._isNormalized = False

            if isInitNow:
                if isinstance(data, str):
                    self.setDataFromFile(data)
                self.setYColumn(yColumn)
                self.setXColumns(xColumns)
                self.setTrainRange(trainRange)
                self.setValidRange(validRange)
                self.setTestRange(testRange)
                if isNormalize:
                    self.normalizeDataUsingTrain()
            else:
                self.xColumns = xColumns
                self.yColumn = yColumn
                self.trainRange = trainRange
                self.validRange = validRange
                self.testRange = testRange

        def setDataFromFile(self, dataFile=None):
            if dataFile is None:
                dataFile = self.dataFile
            assert isinstance(dataFile, str)
            if dataFile.endswith('.npy'):
                data = pd.DataFrame(np.load(dataFile))
            elif dataFile.endswith('.csv'):
                data = pd.read_csv(dataFile, index_col=0, )
            else:
                data = pd.read_pickle(dataFile)
            self.data = data
            return

        def setXColumns(self, xColumns=None):
            if xColumns is None:
                xColumns = self.xColumns

            assert self.data is not None

            if xColumns is None:
                if self.yColumn is not None:
                    xColumns = self.data.columns.drop(self.yColumn)
                else:
                    raise Exception('both x_columns and y_column is None')
            elif isinstance(xColumns, str):
                if 'data' in xColumns and 'self.data' not in xColumns:
                    xColumns = xColumns.replace('data', 'self.data')
                xColumns = eval(xColumns)
            else:
                if isinstance(xColumns[0], int):
                    xColumns = self.data.columns[xColumns]

            for x in xColumns:
                assert x in self.data.columns

            self.xColumns = xColumns
            return

        def setYColumn(self, yColumn=None):

            if yColumn is None:
                yColumn = self.yColumn

            assert self.data is not None

            if isinstance(yColumn, int):
                yColumn = self.data.columns[yColumn]
            elif isinstance(yColumn, str):
                if yColumn in self.data.columns:
                    pass
                else:
                    if 'data' in yColumn and 'self.data' not in yColumn:
                        yColumn = yColumn.replace('data', 'self.data')
                    yColumn = eval(yColumn)
            else:
                raise Exception('Unknown type of y_column')

            self.yColumn = yColumn
            return

        def setTrainRange(self, trainRange=None):
            if trainRange is None:
                trainRange = self.trainRange
            assert self.data is not None
            self.trainRange = self._getRange(trainRange)

        def setValidRange(self, validRange):
            if validRange is None:
                validRange = self.validRange
            assert self.data is not None
            self.validRange = self._getRange(validRange)

        def setTestRange(self, testRange):
            if testRange is None:
                testRange = self.testRange
            assert self.data is not None
            self.testRange = self._getRange(testRange)

        def _getRange(self, dataRange):
            assert len(dataRange) == 2, 'Wrong data range length'
            assert dataRange[0] <= dataRange[1], 'Wrong data range'
            if isinstance(dataRange[0], int) and isinstance(dataRange[1], int):
                return [dataRange[0], dataRange[1]]
            else:
                assert 0 <= dataRange[0] <= 1
                assert 0 <= dataRange[1] <= 1
                n = self.data.shape[0]
                return [int(round(n * dataRange[0])), int(round(n * dataRange[1]))]

        def normalizeDataUsingTrain(self, train_range=None):
            assert not self._isNormalized
            if train_range is not None:
                self.setTrainRange(train_range)
            assert self.data is not None
            data_train = self.data.iloc[self.trainRange[0]:self.trainRange[1], :]
            assert len(data_train.shape) == 2
            self._dataTrainMean = data_train.mean()
            self._dataTrainStd = data_train.std()
            self.data = (self.data - self._dataTrainMean) / self._dataTrainStd
            self._isNormalized = True

        def __str__(self):
            n_blanks, n_params = 0, 20
            string = '[Main] data description:' + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'data_file') + str(
                self.dataFile) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'data_shape') + str(
                self.data.shape) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'x_columns') + str(list(
                self.xColumns)) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'y_column') + str(
                self.yColumn) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'train_range') + str(
                self.trainRange) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'valid_range') + str(
                self.validRange) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'test_range') + str(
                self.testRange) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'is_normalize') + str(
                self._isNormalized) + '\n' + ' ' * n_blanks + '%-*s' % (n_params, 'train_sample_gap') + str(
                self.trainSampleGap)
            return string

        def setTrainTest(self, output_type='DataFrame', ):
            self.xTrain = self.data[self.xColumns].iloc[self.trainRange[0]:self.trainRange[1]]
            self.yTrain = self.data[self.yColumn].iloc[self.trainRange[0]:self.trainRange[1]]
            self.xValid = self.data[self.xColumns].iloc[self.validRange[0]:self.validRange[1]]
            self.yValid = self.data[self.yColumn].iloc[self.validRange[0]:self.validRange[1]]
            self.xTest = self.data[self.xColumns].iloc[self.testRange[0]:self.testRange[1]]
            self.yTest = self.data[self.yColumn].iloc[self.testRange[0]:self.testRange[1]]

            self.xTrainSample = self.xTrain.iloc[::self.trainSampleGap, :]
            self.yTrainSample = self.yTrain.iloc[::self.trainSampleGap]
            self.xValidSample = self.xValid.iloc[::self.trainSampleGap, :]
            self.yValidSample = self.yValid.iloc[::self.trainSampleGap]

            if output_type in ('array', 'ndarray', 'Array'):
                self.xTrain = self.xTrain.values
                self.yTrain = self.yTrain.values
                self.xValid = self.xValid.values
                self.yValid = self.yValid.values
                self.xTest = self.xTest.values
                self.yTest = self.yTest.values

                self.xTrainSample = self.xTrainSample.values
                self.yTrainSample = self.yTrainSample.values
                self.xValidSample = self.xValidSample.values
                self.yValidSample = self.yValidSample.values
            elif output_type in ('DataFrame', 'dataframe', 'dataFrame'):
                pass
            else:
                raise Exception('Unknown type of output_type %s' % (output_type,))