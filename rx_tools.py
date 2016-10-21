import sys
import time
import os
import datetime
import re
import numpy as np
import pandas as pd
import scipy.stats as stat
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LassoLars
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

        class Log(object):

            def __init__(self, file_format='log/%T', is_to_console=True):
                folder = os.path.split(file_format)[0]
                if not folder == '':
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                if '%T' in file_format:
                    time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                    file_format.replace('%T', time_str)
                elif '%D' in file_format:
                    time_str = datetime.datetime.now().strftime('%Y-%m-%d')
                    file_format.replace('%D', time_str)
                self.file_name = file_format
                self.is_to_console = is_to_console

            def start(self):
                self.log_obj = self.PrintLogObject(self.file_name, self.is_to_console)
                self.log_obj.start()

            def close(self):
                self.log_obj.close()

            class PrintLogObject(object):
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


class RxTools(RxToolsBasic):
    """
    Some useful tools from FRX.
    """

    class StatisticTest(object):
        """
        normality test: JB test
        auto-correlation test: Box test
        """

        @staticmethod
        def jb_test(series, level=0.05, is_print=True):
            """
            output: (is_h0_true, p_value, jb_stat, critical value)
            """
            series = series[~np.isnan(series)]
            if len(series) < 100:
                print 'Warning(in JB test): data length: %d' % (len(series),)
            skew = stat.skew(series)
            kurt = stat.kurtosis(series)
            n = len(series)
            jb = (n - 1) * (skew ** 2 + kurt ** 2 / 4) / 6
            p_value = 1 - stat.chi2.cdf(jb, 2)
            cv = stat.chi2.ppf(1 - level, 2)
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
            p_value = stat.chi2.sf(q_stat, lag)
            cv = stat.chi2.ppf(1 - level, lag)
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
            p_value = 2 * (1 - stat.t.cdf(np.abs(t_value), len(dif)))
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
            cv = stat.norm.ppf(1 - level / 2)
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

    class StatisticTools(object):

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

    class OneDAnalysisFunctions(object):

        DEFAULT_MIN_LENGTH = 10

        @staticmethod
        def one_d_check(*args, **kwargs):
            if isinstance(args[-1], int):
                min_length = args[-1]
                args = args[:-1]
            elif 'min_length' in kwargs:
                min_length = kwargs['min_length']
            else:
                min_length = RxTools.OneDAnalysisFunctions.DEFAULT_MIN_LENGTH
            args = [np.array(i).ravel() for i in args]
            length = len(args[0])
            valid = np.full(length, True, np.bool)
            for i in args:
                assert len(i) == length
                valid &= (~np.isnan(i))
            assert np.sum(valid) >= min_length
            args = [i[valid] for i in args]
            return args

        @staticmethod
        def calc_outr2(y, y_hat):
            y, y_hat = RxTools.OneDAnalysisFunctions.one_d_check(y, y_hat)
            y_mean = np.mean(y)
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y_mean) ** 2)

        @staticmethod
        def calc_top_mean(y, y_hat, top_percentage=0.05, top_type='top'):
            y, y_hat = RxTools.OneDAnalysisFunctions.one_d_check(y, y_hat)
            top_num = int(round(len(y) * top_percentage))
            assert top_num > 0, 'Too short data!'
            if top_type in ('top', 't', 'TOP', 'T', 'Top'):
                args = np.argsort(y_hat)[-top_num:]
            elif top_type in ('bottom', 'b', 'BOTTOM', 'B', 'Bottom'):
                args = np.argsort(y_hat)[:top_num]
            else:
                raise Exception('Unknown top_type!')
            return np.mean(y[args])

        @staticmethod
        def calc_stats(x):
            """

            Parameters:
            -------------------------------------
            x: list or 1d array

            Output:
            -------------------------------------
            tuple : length 5
                sharpe, return, std, max_dd, string_to_print

            """
            x = RxTools.OneDAnalysisFunctions.one_d_check(x)
            days_year = 250
            ret = np.mean(x) * days_year
            vol = np.std(x) * np.sqrt(days_year)
            sr = ret / vol
            x_cumsum = np.cumsum(x)
            x_cummax = np.maximum.accumulate(x_cumsum)
            dd = max(x_cummax - x_cumsum)
            str_pre = 'sr=%.2f, ret=%.2f, vol=%.2f, dd=%.2f' % (sr, ret, vol, dd)
            return sr, ret, vol, dd, str_pre

        @staticmethod
        @RxToolsBasic.Wrappers.save_fig_wrapper
        def plot_quantile(x, y, num=20, is_reg=True):
            x, y = RxTools.OneDAnalysisFunctions.one_d_check(x, y, num)
            x_arg = np.argsort(x)
            x_new = x[x_arg]
            y_new = y[x_arg]
            data_length = len(x)
            x_mean = np.array(
                [np.mean(x_new[i * (data_length / num):(i + 1) * (data_length / num)]) for i in range(num)])
            y_mean = np.array(
                [np.mean(y_new[i * (data_length / num):(i + 1) * (data_length / num)]) for i in range(num)])
            plt.scatter(x_mean, y_mean)
            plt.title('qq plot')
            if is_reg:
                model = sm.OLS(y_mean, sm.add_constant(x_mean)).fit()
                params = model.params
                y_hat = x_mean * params[1] + params[0]
                plt.plot(x_mean, y_hat)

        @staticmethod
        def plot_distribution(x, bins='auto', distribution=''):
            """
            input:  bins : 'auto' or int
                    distribution: '' or 'norm'
            """
            x = RxTools.OneDAnalysisFunctions.one_d_check(x, 10)
            assert bins == 'auto' or type(bins) == int
            if bins == 'auto':
                bins = int(np.sqrt(len(x))) if len(x) >= 100 else (10 if len(x) >= 10 else len(x))
            all_distribution_list = ['', 'norm']
            assert distribution in all_distribution_list
            normed = False if distribution == '' else True
            plt.hist(x, bins=bins, normed=normed)
            plt.title('distribution hist')
            if distribution == 'norm':
                norm_mean = np.mean(x)
                norm_std = np.std(x)
                xlist_0 = [i / 1000. for i in range(1001)]
                xlist_1 = [stat.norm.ppf(i) for i in xlist_0]
                ylist = [stat.norm.pdf(i) for i in xlist_1]
                xlist = norm_mean + norm_std * np.array(xlist_1)
                plt.plot(xlist, ylist)
            plt.show()

        @staticmethod
        def calc_top_batch_mean(y, y_hat, top_percentage=0.025, top_gap=0.0025, top_type='both'):
            y = np.array(y).ravel()
            y_hat = np.array(y_hat).ravel()
            batch_num = int(top_percentage / top_gap)
            batch_size = int(round(len(y) * top_gap))
            assert batch_size > 0, 'Too small batch! %d' % (batch_size,)
            if top_type in ('top', 't', 'TOP', 'T', 'Top'):
                mean_list = []
                for b in range(batch_num):
                    if b == 0:
                        mean_list.append(np.mean(y[np.argsort(y_hat)[-batch_size:]]))
                    else:
                        mean_list.append(np.mean(y[np.argsort(y_hat)[-batch_size * (b + 1):-batch_size * b]]))
            elif top_type in ('bottom', 'b', 'BOTTOM', 'B', 'Bottom'):
                mean_list = [np.mean(y[np.argsort(y_hat)[batch_size * b:batch_size * (b + 1)]]) for b in range(batch_num)]
            elif top_type == 'both':
                mean_list = []
                for b in range(batch_num):
                    if b == 0:
                        mean_list.append(np.mean(y[np.argsort(y_hat)[-batch_size:]] -
                                                 y[np.argsort(y_hat)[batch_size * b:batch_size * (b + 1)]]) / 2.)
                    else:
                        mean_list.append(np.mean(y[np.argsort(y_hat)[-batch_size * (b + 1):-batch_size * b]] -
                                                 y[np.argsort(y_hat)[batch_size * b:batch_size * (b + 1)]]) / 2.)
            else:
                raise Exception('Unknown top_type!')
            return mean_list

    class TwoDAnalysisFunctions(object):

        @staticmethod
        def calc_turnover(x, axis=0, base='yesterday'):
            """

            Parameters:
            -------------------------------------
            x: 2d array
            axis: 0 or 1, default 0
                when 0, every column is a sample, we cal turnover between columns
            base: 'yesterday' or 'average' or (int or float), default 'yesterday'
                when (int or float), abs / base
                when 'yesterday', abs / sum(abs(yesterday))
                when 'average', abs / average(sum(abs(every_day)))
            """
            x = np.array(x)
            x = np.nan_to_num(x)
            if axis == 1:
                x = x.T
            turnover = np.full(x.shape[1] - 1, np.nan)
            for i in range(x.shape[1] - 1):
                if isinstance(base, int) or isinstance(base, float):
                    pass
                elif base == 'yesterday':
                    base = np.sum(np.abs(x[:, i]))
                elif base == 'average':
                    base = np.mean(np.sum(np.abs(x), axis=0))
                elif base == 'normal':
                    pass
                else:
                    raise Exception('Unknown base!')
                turnover[i] = np.sum(np.abs(x[:, i + 1] - x[:, i])) / base
            return turnover

        @staticmethod
        def calc_corr(x, y, axis=0, min_calc_length=10):
            if axis == 1:
                x, y = x.T, y.T
            corr = np.full(x.shape[1], np.nan)
            for i in range(x.shape[1]):
                valid = (~np.isnan(x[:, i])) & (~np.isnan(y[:, i]))
                if np.sum(valid) <= min_calc_length:
                    continue
                corr[i] = np.corrcoef(x[valid, i], y[valid, i])[0, 1]
            return corr

        @staticmethod
        def calc_beta(x, y, axis=0, min_calc_length=10):
            if axis == 1:
                x, y = x.T, y.T
            beta = np.full(x.shape[1], np.nan)
            for i in range(x.shape[1]):
                valid = (~np.isnan(x[:, i])) & (~np.isnan(y[:, i]))
                if np.sum(valid) <= min_calc_length:
                    continue
                reg_x, reg_y = x[valid, i], y[valid, i]
                try:
                    res = sm.OLS(reg_y, sm.add_constant(reg_x)).fit()
                    beta[i] = res.params[1]
                except:
                    continue
            return

    class VariableSelection(object):

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

            def select(self, x, y, x_columns=None):
                x, y = self._check_data(x, y)
                if x_columns is None:
                    x_columns = list(x.columns)
                else:
                    x_columns = list(x_columns)
                    for x_column in x_columns:
                        assert x_column in x.columns
                return self._select(x, y, x_columns)

            def _select(self, x, y, x_columns):
                raise NotImplementedError

        class RemoveAllConst(AbstractSelection):

            def __init__(self, is_print=False):
                self.is_print = is_print

            def _select(self, x, y, x_columns):
                if self.is_print:
                    print '[Remove All Const] selecting ...'
                for x_column in x_columns:
                    x_single = x[x_column].values
                    if len(sm.add_constant(x_single).shape) == 1:
                        x_columns.remove(x_column)
                        if self.is_print:
                            print '[Remove All Const] %d remain, remove %s, all constant' \
                                  % (len(x_columns), x_column, )
                return x_columns

        class BackwardSingleP(AbstractSelection):
            def __init__(self, p_threshold=0.05, is_print=False):
                self.p_threshold = p_threshold
                self.is_print = is_print

            def _select(self, x, y, x_columns):
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

            def _select(self, x, y, x_columns):
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

            def _select(self, x, y, x_columns):
                if self.is_print:
                    print '[Select Margin T] selecting ... %d remain' % (len(x_columns), )
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

            def _select(self, x, y, x_columns):
                if self.is_print:
                    print '[Select Margin F] selecting ... %d remain' % (len(x_columns),)
                    print '[Select Margin F] group size: %d' % (self.group_size,)
                    print '[Select Margin F] F P-value: %.4f, min num of var: %d' % (self.f_p_value, self.n_min)
                while len(x_columns) > self.n_min:
                    bench = sm.OLS(y, x[x_columns]).fit()
                    p_values_sorted = bench.pvalues.sort_values()
                    group = p_values_sorted[-self.group_size:]
                    print '[Select Margin F]', list(group.index)
                    print '[Select Margin F]', list(group.values)
                    if self.f_p_value == 0.0:
                        for x_column in list(reversed(list(group.index))):
                            x_columns.remove(x_column)
                            if self.is_print:
                                print '[Select Margin F] %d remain, remove %s, f p-value: %.4f' \
                                  % (len(x_columns), x_column, f_value)
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

    class Protocols(object):
        """
        every protocol should INHERIT abstract class AbstractProtocal.
        data all saved as dict.


        INHERITING:
        You should rewrite these:
        1. _item_help
        2. default_save_folder
        3. _checkData

        ATTENTION: 'name' and 'save_folder' should not in your keys


        USING:
        if you want to load, there are 2 methods
        1. when init, give name and save folder
        2. call protocol.load(name, save_folder)

        if you want to add item, there are 2 methods
        1. when init, DON'T give name, but give keys and values
        2. call protocol.set_items(key, value)

        if you want to save
        1. call protocol.save(name, save_folder)

        """

        class AbstractProtocol(object):

            _items_help = {}
            default_save_folder = None

            def __init__(self, **kwargs):
                self.name = None
                self.save_folder = None
                if kwargs.get('name'):
                    self.load(kwargs.get('name'), kwargs.get('save_folder'))
                else:
                    self._items_dict = dict()
                    for k, value in kwargs.items():
                        if k in self._items_help:
                            self._items_dict[k] = value
                        else:
                            print 'WARNING: %s is no use!' % (k,)

            @classmethod
            def print_help(cls):
                print "*" * 50
                for key in cls._items_help:
                    print key, ' :\n\t', cls._items_help[key]
                print "*" * 50
                return

            def print_items(self, is_print_value=False):
                for key in self._items_dict:
                    if is_print_value:
                        print key, ' :\n', self._items_dict[key]
                    else:
                        print key
                return

            def set_item(self, item, value):
                self._items_dict[item] = value

            def get_item(self, item):

                try:
                    return self._items_dict[item]
                except KeyError:
                    name = 'Current object' if self.name is None else self.name
                    print 'Warning: ' + name + ' has no item %s, return None' % (item,)
                    return

            def _check_data(self):
                pass

            def save(self, name, save_folder=None):
                self._check_data()
                self.name = name
                save_folder = self.default_save_folder if save_folder is None else save_folder
                self.save_folder = save_folder
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                path = os.path.join(save_folder, name)
                pd.to_pickle(self._items_dict, path)

            def load(self, name, save_folder=None):
                save_folder = self.default_save_folder if save_folder is None else save_folder
                path = os.path.join(save_folder, name)
                self.name = name
                self.save_folder = save_folder
                self._items_dict = pd.read_pickle(path)

            def update(self):
                assert self.name is not None and self.save_folder is not None
                self.save(self.name, self.save_folder)

        class NeuralNetworkProtocol(AbstractProtocol):

            _items_help = {'para_in': 'input parameters',
                           'para': 'output parameters',
                           'call_code': 'call code',
                           'x_test': 'rt',
                           'y_test': 'rt',
                           'x_train': 'rt',
                           'y_train': 'rt',
                           'x_valid': 'validation x',
                           'y_valid': 'validation y',
                           'r2_is': 'in sample r2',
                           'r2_os': 'out sample r2',
                           'r2_valid': 'validation sample r2',
                           'y_hat_is': 'in sample y^',
                           'y_hat_os': 'out sample y^',
                           'y_hat_valid': 'validation y^',
                           'layer_code': 'layer_code',
                           }
            default_save_folder = ''

    class LogAnalysis(object):

        @staticmethod
        def log_analysis_single_re(log_str, expression, keys, functions=None):
            if isinstance(keys, str):
                keys = [keys]
            if functions is not None:
                assert len(keys) == len(functions)
            mappings = re.findall(expression, log_str)
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

    class DataPrepare(object):
        def __init__(self, data, x_columns=None, y_column='y',
                     train_range=(0.4, 1.), valid_range=(0.2, 0.4), test_range=(0., 0.2),
                     is_normalize=False, train_sample_gap=1, init_now=False, ):

            # init data and data_file
            if isinstance(data, str):
                self.data_file = data
                self.data = None
            else:
                self.data_file = None
                if isinstance(data, np.ndarray) or isinstance(data, list):
                    self.data = pd.DataFrame(data)
                elif isinstance(data, pd.DataFrame):
                    self.data = data
                else:
                    raise Exception('Unknown type of data!')

            self.train_sample_gap = train_sample_gap
            self._is_normalized = False

            if init_now:
                if isinstance(data, str):
                    self.set_data_from_file(data)
                self.set_y_column(y_column)
                self.set_x_columns(x_columns)
                self.set_train_range(train_range)
                self.set_valid_range(valid_range)
                self.set_test_range(test_range)
                if is_normalize:
                    self.normalize_data_using_train()
            else:
                self.x_columns = x_columns
                self.y_column = y_column
                self.train_range = train_range
                self.valid_range = valid_range
                self.test_range = test_range

        def set_data_from_file(self, data_file=None):
            if data_file is None:
                data_file = self.data_file
            assert isinstance(data_file, str)
            if data_file.endswith('.npy'):
                data = pd.DataFrame(np.load(data_file))
            elif data_file.endswith('.csv'):
                data = pd.read_csv(data_file, index_col=0, )
            else:
                data = pd.read_pickle(data_file)
            self.data = data
            return


        def set_x_columns(self, x_columns=None):
            if x_columns is None:
                x_columns = self.x_columns

            assert self.data is not None

            if x_columns is None:
                if self.y_column is not None:
                    x_columns = self.data.columns.drop(self.y_column)
                else:
                    raise Exception('both x_columns and y_column is None')
            elif isinstance(x_columns, str):

                if 'data' in x_columns and 'self.data' not in x_columns:
                    x_columns = x_columns.replace('data', 'self.data')

                x_columns = eval(x_columns)

            else:
                if isinstance(x_columns[0], int):
                    x_columns = self.data.columns[x_columns]

                for x in x_columns:
                    assert x in self.data.columns
            self.x_columns = x_columns
            return

        def set_y_column(self, y_column=None):

            if y_column is None:
                y_column = self.y_column

            assert self.data is not None

            if isinstance(y_column, int):
                y_column = self.data.columns[y_column]
            elif isinstance(y_column, str):
                if y_column in self.data.columns:
                    pass
                else:
                    if 'data' in y_column and 'self.data' not in y_column:
                        y_column = y_column.replace('data', 'self.data')
                    y_column = eval(y_column)
            else:
                raise Exception('Unknown type of y_column')

            self.y_column = y_column
            return

        def set_train_range(self, train_range=None):
            if train_range is None:
                train_range = self.train_range
            assert self.data is not None
            self.train_range = self._get_range(train_range)

        def set_valid_range(self, valid_range):
            if valid_range is None:
                valid_range = self.valid_range
            assert self.data is not None
            self.valid_range = self._get_range(valid_range)

        def set_test_range(self, test_range):
            if test_range is None:
                test_range = self.test_range
            assert self.data is not None
            self.test_range = self._get_range(test_range)

        def _get_range(self, data_range):
            assert len(data_range) == 2, 'Wrong data range length'
            assert data_range[0] <= data_range[1], 'Wrong data range'
            if isinstance(data_range[0], int) and isinstance(data_range[1], int):
                return [data_range[0], data_range[1]]
            else:
                assert 0 <= data_range[0] <= 1
                assert 0 <= data_range[1] <= 1
                n = self.data.shape[0]
                return [int(round(n * data_range[0])), int(round(n * data_range[1]))]

        def normalize_data_using_train(self, train_range=None):
            assert not self._is_normalized
            if train_range is not None:
                self.set_train_range(train_range)
            assert self.data is not None
            data_train = self.data.iloc[self.train_range[0]:self.train_range[1], :]
            assert len(data_train.shape) == 2
            self._data_train_mean = data_train.mean()
            self._data_train_std = data_train.std()
            self.data = (self.data - self._data_train_mean) / self._data_train_std
            self._is_normalized = True

        def __str__(self):
            n_blanks, n_params = 0, 20
            string = 'Data description:' + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'data_file') + str(self.data_file) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'data_shape') + str(self.data.shape) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'x_columns') + str(list(self.x_columns)) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'y_column') + str(self.y_column) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'train_range') + str(self.train_range) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'train_range_index') + str(self.data.index[self.train_range[0]]) + '   ' + str(self.data.index[self.train_range[1] - 1]) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'valid_range') + str(self.valid_range) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'valid_range_index') + str(self.data.index[self.valid_range[0]]) + '   ' + str(self.data.index[self.valid_range[1] - 1]) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'test_range') + str(self.test_range) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'test_range_index') + str(self.data.index[self.test_range[0]]) + '   ' + str(self.data.index[self.test_range[1] - 1]) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'is_normalize') + str(self._is_normalized) + '\n' + \
                     ' ' * n_blanks + '%-*s' % (n_params, 'train_sample_gap') + str(self.train_sample_gap)
            return string

        def set_train_test(self):
            self.x_train = self.data[self.x_columns].iloc[self.train_range[0]:self.train_range[1]].values
            self.y_train = self.data[self.y_column].iloc[self.train_range[0]:self.train_range[1]].values
            self.x_valid = self.data[self.x_columns].iloc[self.valid_range[0]:self.valid_range[1]].values
            self.y_valid = self.data[self.y_column].iloc[self.valid_range[0]:self.valid_range[1]].values
            self.x_test = self.data[self.x_columns].iloc[self.test_range[0]:self.test_range[1]].values
            self.y_test = self.data[self.y_column].iloc[self.test_range[0]:self.test_range[1]].values

            self.x_train_sample = self.x_train[::self.train_sample_gap, :]
            self.y_train_sample = self.y_train[::self.train_sample_gap]
            self.x_valid_sample = self.x_valid[::self.train_sample_gap, :]
            self.y_valid_sample = self.y_valid[::self.train_sample_gap]
