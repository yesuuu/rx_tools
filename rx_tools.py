import time
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stat
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model


class RxToolsBasic(object):
    class Decorators(object):

        @staticmethod
        def deco_save_fig(default_save_file):
            """
            A decorator to save pictures
            in func, you should plot a single figure, neither show nor save_fig.

            Parameters
            -------------------------------------
            default_save_file:
                    only file name of picture, not including save path

            Notes
            -------------------------------------
            in the function, 'save_path' and 'save_file' should not be kwargs

            'save_path' means the file path you want to save pictures to
            'save_file' means the file name
            """

            def save_fig(func):
                def new_func(*args, **kwargs):
                    if 'save_path' in kwargs:
                        save_path = kwargs['save_path']
                        del kwargs['save_path']
                    else:
                        save_path = None

                    if 'save_file' in kwargs:
                        save_file = kwargs['save_file']
                        del kwargs['save_file']
                    else:
                        save_file = default_save_file
                    func(*args, **kwargs)
                    if len(plt.get_fignums()) != 1:
                        raise Exception('plt model has more than one picture')
                    if not save_path:
                        plt.show()
                    else:
                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)
                        plt.savefig(os.path.join(save_path, save_file))
                        plt.close()

                return new_func

            return save_fig

        @staticmethod
        def deco_calc_time(func):
            def new_func(*args, **kwargs):
                time1 = time.clock()
                result = func(*args, **kwargs)
                time2 = time.clock()
                diff_time = time2 - time1
                print 'exec time: %.8f s' % (diff_time,)
                return result

            return new_func

    class Others(object):

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
        @RxToolsBasic.Decorators.deco_save_fig('qq plot')
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

    class RegressionTools(object):

        @staticmethod
        def find_lasso_para(x, y, paras=None, start_exp=-10, end_exp=-10, ):
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
                tmp_model = linear_model.LassoLars(alpha=para)
                tmp_model.fit(sm.add_constant(x), y)
                tmp_coef = tmp_model.coef_
                variable_num.append(np.sum(tmp_coef != 0))
                params.append(tmp_coef)
            return paras, variable_num, params

    class VariableSelection(object):

        @staticmethod
        def _check_data(x, y):
            x = np.array(x)
            assert len(x.shape) == 2
            y = np.array(y).ravel()
            assert x.shape[0] == len(y)
            if len(y) < 100:
                print 'Warning: data length %d too small ' % (len(y),)
            return x, y

        @staticmethod
        def backward_selection_r2(x, y, p_value=0.05, threshold=0.001, is_print=True):
            x, y = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                x_reg = sm.add_constant(x[:, i])
                if len(x_reg.shape) == 1:
                    select[i] = False
                    if is_print:
                        print '%d dropped, all const' % (i,)
                    break
                model = sm.OLS(y, x_reg).fit()
                if model.pvalues[1] > p_value:
                    select[i] = False
                    if is_print:
                        print '%d dropped, pvalue %.4f' % (i, model.pvalues[1])
            in_model_num = int(np.sum(select))
            while in_model_num > 1:
                if is_print:
                    print 'In model num: %d' % (in_model_num,)
                select_index = [i for i in range(x.shape[1]) if select[i] == True]
                bench = sm.OLS(y, sm.add_constant(x[:, select_index])).fit()
                group_size = in_model_num / 30 + 1 if in_model_num < 300 else 10
                if is_print:
                    print 'group size: %d' % (group_size,)
                group_num = in_model_num / group_size + 1
                select_index_group = RxToolsBasic.Others.divide_into_group(select_index, group_num=group_num)
                assert len(select_index_group) == group_num
                adj_r2 = np.zeros(group_num)
                for i in range(group_num):
                    if is_print:
                        print 'calc adj_r2:', i
                    tmp_select = [j for j in select_index if j not in select_index_group[i]]
                    tmp_res = sm.OLS(y, sm.add_constant(x[:, tmp_select])).fit()
                    adj_r2[i] = tmp_res.rsquared_adj
                rsquared_loss = bench.rsquared_adj - adj_r2
                max_del_num = (group_num / 20 + 1) if (group_num <= 100) else (group_num / 10)
                loss_min_arg = rsquared_loss.argsort()[:max_del_num]
                for i in range(max_del_num):
                    if rsquared_loss[loss_min_arg[i]] < threshold:
                        select[np.array(select_index_group[loss_min_arg[i]])] = False
                        if is_print:
                            print '\t', select_index_group[loss_min_arg[i]], ' dropped, rsquared loss %.2fE-3' % (
                                rsquared_loss[loss_min_arg[i]] * 1e3,)
                    else:
                        break
                in_model_num_new = int(np.sum(select))
                if in_model_num == in_model_num_new:
                    break
                else:
                    in_model_num = in_model_num_new
            return select

        @staticmethod
        def backward_selection_t(x, y, p_value=0.05, t_threshold=2.0, is_print=True):
            x, y = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                x_reg = sm.add_constant(x[:, i])
                if len(x_reg.shape) == 1:
                    select[i] = False
                    if is_print:
                        print '%d dropped, all const' % (i,)
                    break
                model = sm.OLS(y, x_reg).fit()
                if model.pvalues[1] > p_value:
                    select[i] = False
                    if is_print:
                        print '%d dropped, pvalue %.4f' % (i, model.pvalues[1])
            in_model_num = int(np.sum(select))
            while in_model_num > 1:
                if is_print:
                    print 'In model num: %d' % (in_model_num,)
                select_index = [i for i in range(x.shape[1]) if select[i] == True]
                bench = sm.OLS(y, sm.add_constant(x[:, select])).fit()
                t_values = bench.tvalues[1:]
                for i in range(len(t_values)):
                    if np.abs(t_values[i]) < t_threshold:
                        select[select_index[i]] = False
                        if is_print:
                            print '%d dropped, t value: %0.4f' % (select_index[i], t_values[i])
                        break
                in_model_num_new = int(np.sum(select))
                if in_model_num == in_model_num_new:
                    break
                else:
                    in_model_num = in_model_num_new
            return select

        @staticmethod
        def backward_selection_t2(x, y, t_threshold=2.0, is_print=True):
            x, y = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                x_reg = sm.add_constant(x[:, i])
                if len(x_reg.shape) == 1:
                    select[i] = False
                    if is_print:
                        print '%d dropped, all const' % (i,)
                    continue
            in_model_num = int(np.sum(select))
            while in_model_num > 1:
                if is_print:
                    print 'In model num: %d' % (in_model_num,)
                select_index = [i for i in range(x.shape[1]) if select[i] == True]
                bench = sm.OLS(y, sm.add_constant(x[:, select])).fit()
                t_values = bench.tvalues[1:]
                t_values_abs = np.abs(t_values)
                t_values_min_arg = np.argmin(t_values_abs)
                if t_values_abs[t_values_min_arg] < t_threshold:
                    select[select_index[t_values_min_arg]] = False
                    in_model_num -= 1
                    if is_print:
                        print '%d dropped, t value: %0.4f' % (select_index[t_values_min_arg],
                                                              t_values[t_values_min_arg])
                else:
                    break
            return select

        @staticmethod
        def backward_selection_f(x, y, p_value=1.0, group_size_list=(5, 1),
                                 f_p_value_list=(0.5, 0.05), is_print=True):
            assert len(group_size_list) == len(f_p_value_list)
            (x, y) = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                x_reg = sm.add_constant(x[:, i])
                if len(x_reg.shape) == 1:
                    select[i] = False
                    if is_print:
                        print '%d dropped, all const' % (i,)
                    continue
                model = sm.OLS(y, x_reg).fit()
                if model.pvalues[1] > p_value:
                    select[i] = False
                    if is_print:
                        print '%d dropped, pvalue %.4f' % (i, model.pvalues[1])
            for group_size, f_p_value in zip(group_size_list, f_p_value_list):
                if is_print:
                    print 'group size:', group_size
                while True:
                    if is_print:
                        print 'in_model_num', int(np.sum(select))
                    select_index = [nn for nn in range(x.shape[1]) if select[nn] == True]
                    bench = sm.OLS(y, x[:, select_index]).fit()
                    p_values = bench.pvalues
                    p_values_argsort = np.flipud(np.argsort(p_values))
                    group = p_values_argsort[:group_size]
                    print p_values[group]
                    restricted_index = [j for j in select_index if j not in np.array(select_index)[group]]
                    restricted_model = sm.OLS(y, x[:, restricted_index]).fit()
                    tmpres = bench.compare_f_test(restricted_model)
                    print tmpres
                    f_value = tmpres[1]
                    if f_value > f_p_value:
                        select[np.array(np.array([select_index[k] for k in group]))] = False
                        if is_print:
                            print np.array(select_index)[group], ' dropped, p_value %.4f' % (f_value,)
                    else:
                        break
            return select

        @staticmethod
        def p_value_selection(x, y, p_value=0.03, is_print=True):
            x, y = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                x_reg = sm.add_constant(x[:, i])
                if len(x_reg.shape) == 1:
                    select[i] = False
                    if is_print:
                        print '%d dropped, all const' % (i,)
                    break
                model = sm.OLS(y, x_reg).fit()
                if model.pvalues[1] > p_value:
                    select[i] = False
                    if is_print:
                        print '%d dropped, pvalue %.4f' % (i, model.pvalues[1])
            return select

        @staticmethod
        def corr_selection(x, y, threshold):
            x, y = RxTools.VariableSelection._check_data(x, y)
            select = np.full(x.shape[1], True, bool)
            for i in range(x.shape[1]):
                if np.abs(np.corrcoef(x[:, i], y)[0, 1]) < threshold:
                    select[i] = False
            return select

    class Protocols(object):
        """
        every protocol should INHERIT abstract class AbstractProtocal.
        data all saved as dict.


        INHERITING:
        You should rewrite these:
        1. _item_help
        2. default_save_folder
        3. _check_data

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
                if kwargs.get('name'):
                    self.load(kwargs.get('name'), kwargs.get('save_folder'))
                else:
                    self._items_dict = dict()
                    for k, value in kwargs.items():
                        if k in self._items_help:
                            self._items_dict[k] = value
                        else:
                            print 'WARNING: %s is no use!' % (k, )

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
                    warnings.warn('No item %s, return None' % (item,))
                    return None

            def _check_data(self):
                pass

            def save(self, name, save_folder=None):
                self._check_data()
                self.name = name
                save_folder = self.default_save_folder if save_folder is None else save_folder
                if not os.path.isdir(save_folder):
                    os.makedirs(save_folder)
                path = os.path.join(save_folder, name)
                pd.to_pickle(self._items_dict, path)

            def load(self, name, save_folder=None):
                save_folder = self.default_save_folder if save_folder is None else save_folder
                path = os.path.join(save_folder, name)
                self.name = name
                self._items_dict = pd.read_pickle(path)

        class NeuralNetworkProtocol(AbstractProtocol):

            _items_help = {'para_in': 'input parameters',
                           'para': 'output parameters',
                           'call_code': 'call code',
                           'x_test': 'rt',
                           'y_test': 'rt',
                           'x_train': 'rt',
                           'y_train': 'rt',
                           'r2_is': 'in sample r2',
                           'r2_os': 'out sample r2',
                           'y_hat_is': 'rt',
                           'y_hat_os': 'rt',
                           }
            default_save_folder = '/home/fanruoxin/fit_mlp/results2'
