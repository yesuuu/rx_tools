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
import seaborn

matplotlib.use("Qt4Agg")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 15)


class RxModeling(object):

    class DataPrepare(object):

        class CrossValidationMethod(object):
            """
            output: [(train_idx1, test_idx1),
                     (train_idx2, test_idx2),
                     ...
                     (train_idxn, test_idxn),]
            """

            @staticmethod
            def k_folds(data_length, n_folders, random_state=None):
                data_idx = range(data_length)
                if random_state is not None:
                    if isinstance(random_state, int):
                        random.seed(random_state)
                    random.shuffle(data_idx, random=random_state)
                group_size_small = data_length / n_folders
                group_num_big = (data_length % n_folders)
                nums = [(group_size_small + 1 if i < group_num_big else group_size_small)
                        for i in range(n_folders)]
                nums.insert(0, 0)
                indexs = list(np.cumsum(nums))
                idx_list = []
                for k in range(n_folders):
                    test_idx = data_idx[indexs[k]: indexs[k+1]]
                    train_idx = data_idx[:indexs[k]] + data_idx[indexs[k+1]:]
                    idx_list.append((train_idx, test_idx, ))
                return idx_list

        class GetData(object):

            @staticmethod
            def from_file(data_file):
                assert isinstance(data_file, str)
                if data_file.endswith('.npy'):
                    data = pd.DataFrame(np.load(data_file))
                elif data_file.endswith('.csv'):
                    data = pd.read_csv(data_file, index_col=0, )
                else:
                    data = pd.read_pickle(data_file)
                return data

            @staticmethod
            def from_origin(data_file, ask_or_bid='ask', y_length=60):
                assert isinstance(data_file, str) and data_file.endswith('.pic')
                data_dict = pd.read_pickle(data_file)
                data = data_dict['x']
                data_y = data_dict['y'][ask_or_bid].loc[data.index, y_length]
                data['y'] = np.sign(data_y) * np.log(np.abs(data_y) + 1)
                data.dropna(axis=0, inplace=True)
                return data

        def __init__(self, data=None):

            self.data = data
            self.data_file = None

            self.train_range = None
            self.train_idx = None
            self.valid_range = None
            self.valid_idx = None
            self.test_range = None
            self.test_idx = None

            self._data_train_mean = None
            self._data_train_std = None
            self._is_normalized = False

            self.x_train = None
            self.y_train = None
            self.x_valid = None
            self.y_valid = None
            self.x_test = None
            self.y_test = None

            self.x_columns = None
            self.y_column = None

        def set_data(self, data=None, data_file=None):
            self.data = data if data is not None else self.data
            self.data_file = data_file if data_file is not None else self.data_file

        def set_x_columns(self, x_columns=None):
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

                for x_column in x_columns:
                    assert x_column in self.data.columns
            self.x_columns = list(x_columns)
            return

        def add_x_column(self, x_column_name, x_column_data):
            self.data[x_column_name] = x_column_data
            self.x_columns.append(x_column_name)

        def set_y_column(self, y_column):
            if isinstance(y_column, int):
                y_column = self.data.columns[y_column]
            elif isinstance(y_column, str):
                if y_column in self.data.columns:
                    pass
                else:
                    if 'data' in y_column and 'self.data' not in y_column:
                        y_column = y_column.replace('data', 'self.data')
                    y_column = eval(y_column)
                    assert y_column in self.data.columns
            else:
                raise Exception('Unknown type of y_column')
            self.y_column = y_column
            return

        def set_train_idx(self, train_range, range_type=None, sample_gap=1):
            self.train_range, self.train_idx = self._get_idx(train_range, range_type, sample_gap)

        def set_valid_idx(self, valid_range, range_type=None, sample_gap=1):
            self.valid_range, self.valid_idx = self._get_idx(valid_range, range_type, sample_gap)

        def set_test_idx(self, test_range, range_type=None, sample_gap=1):
            self.test_range, self.test_idx = self._get_idx(test_range, range_type, sample_gap)

        def _get_idx(self, data_range, range_type=None, sample_gap=1):
            if not range_type:
                if len(data_range) == 2:
                    range_type = 'range'
                else:
                    range_type = 'index'
            if range_type == 'range':
                data_range = data_range[:2]
                assert data_range[0] <= data_range[1], 'Wrong data range'
                if isinstance(data_range[0], int) and isinstance(data_range[1], int):
                    pass
                else:
                    assert 0 <= data_range[0] <= 1
                    assert 0 <= data_range[1] <= 1
                    n = self.data.shape[0]
                    data_range = [int(round(n * data_range[0])), int(round(n * data_range[1]))]
                data_numidx = range(data_range[0], data_range[1])
            elif range_type == 'index':
                data_numidx = data_range
                data_range = None
            else:
                raise Exception('Unknown range_type: %s' % (str(range_type), ))

            data_range = list(data_range[:2]) + [sample_gap] if data_range is not None else None
            data_idx = self.data.index[data_numidx[::sample_gap]]

            return data_range, data_idx

        def normalize_using_train(self):
            assert not self._is_normalized
            data_train = self.data.loc[self.train_idx, :]
            self._data_train_mean = data_train.mean()
            self._data_train_std = data_train.std()
            self.data = (self.data - self._data_train_mean) / self._data_train_std
            self._is_normalized = True

        def unnormalize(self):
            assert self._is_normalized
            self.data = self.data * self._data_train_std + self._data_train_mean
            self._is_normalized = False
            self._data_train_mean = None
            self._data_train_std = None
            return

        def set_train_test(self, output_type='DataFrame'):
            """
            output_type in ('DataFrame', 'ndarray')
            """
            if self.train_idx is not None:
                self.x_train = self.data.loc[self.train_idx, self.x_columns]
                self.y_train = self.data.loc[self.train_idx, self.y_column]
            if self.valid_idx is not None:
                self.x_valid = self.data.loc[self.valid_idx, self.x_columns]
                self.y_valid = self.data.loc[self.valid_idx, self.y_column]
            if self.test_idx is not None:
                self.x_test = self.data.loc[self.test_idx, self.x_columns]
                self.y_test = self.data.loc[self.test_idx, self.y_column]

            if output_type in ('array', 'ndarray', 'Array'):
                self.x_train = self.x_train.values if self.x_train is not None else None
                self.y_train = self.y_train.values if self.y_train is not None else None
                self.x_valid = self.x_valid.values if self.x_valid is not None else None
                self.y_valid = self.y_valid.values if self.y_valid is not None else None
                self.x_test = self.x_test.values if self.x_test is not None else None
                self.y_test = self.y_test.values if self.y_test is not None else None
            elif output_type in ('DataFrame', 'dataFrame', 'dataframe'):
                pass
            else:
                raise Exception('Unknown type of output_type %s' % (output_type, ))
            return

        def __str__(self):
            n_params = 20
            string = \
                'Data description:' + '\n' + \
                ('' if self.data_file is None else '%-*s' % (n_params, 'data_file') + str(self.data_file) + '\n') + \
                '%-*s' % (n_params, 'data_shape') + str(self.data.shape) + '\n' + \
                '%-*s' % (n_params, 'x_columns') + str(list(self.x_columns)) + '\n' + \
                '%-*s' % (n_params, 'x_num') + str(len(list(self.x_columns))) + '\n' + \
                '%-*s' % (n_params, 'y_column') + str(self.y_column) + '\n' + \
                ('' if self.train_range is None else '%-*s' % (n_params, 'train_range') + str(self.train_range) + '\n' +
                 '%-*s' % (n_params, 'train_range_index') + str(self.data.index[self.train_range[0]]) + '   ' + str(self.data.index[self.train_range[1] - 1]) + '\n') + \
                ('' if self.valid_range is None else '%-*s' % (n_params, 'valid_range') + str(self.valid_range) + '\n' +
                 '%-*s' % (n_params, 'valid_range_index') + str(self.data.index[self.valid_range[0]]) + '   ' + str(self.data.index[self.valid_range[1] - 1]) + '\n') + \
                ('' if self.test_range is None else '%-*s' % (n_params, 'test_range') + str(self.test_range) + '\n' +
                 '%-*s' % (n_params, 'test_range_index') + str(self.data.index[self.test_range[0]]) + '   ' + str(self.data.index[self.test_range[1] - 1]) + '\n') + \
                '%-*s' % (n_params, 'is_normalize') + str(self._is_normalized) + '\n'
            return string

    class Log(object):

        def __init__(self, file_name='log/%T', is_to_console=True):
            self.log_obj = None
            self.file_name = self.reformat_file_name(file_name)
            self.is_to_console = is_to_console

        @staticmethod
        def reformat_file_name(file_name):
            time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
            if '%T' in file_name:
                file_name = file_name.replace('%T', time_str)
            if '%D' in file_name:
                file_name = file_name.replace('%D', time_str.split('T')[0])
            return file_name

        def start(self, is_print=False):
            self.log_obj = self.SavePrint(self.file_name, self.is_to_console)
            self.log_obj.start()
            if is_print:
                print '[log] log starts, to file %s' % (self.file_name, )

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

    class Time(object):
        def __init__(self, is_now=False, is_all=False, is_margin=False):
            self.start_time = None
            self.last_time = None
            self.is_now = is_now
            self.is_all = is_all
            self.is_margin = is_margin

        def show(self):
            now = datetime.datetime.now()
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

    class X(object):

        @staticmethod
        def calc_basic_statistics(x, info=None):
            info = ['mean', 'std', 'skew', 'kurt', 'num', 'num_out2std', 'num_out3std', 'num_out5std', 'num_out10std'] \
                if info is None else info

            x = np.array(x).ravel()

            func_map = {'mean': np.mean,
                        'std': np.std,
                        'skew': stats.skew,
                        'kurt': stats.kurtosis,
                        'num': len,
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

    class YHat(object):

        @staticmethod
        def calc_outr2(y, y_hat):
            y, y_hat = np.array(y), np.array(y_hat)
            y_mean = np.mean(y)
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y_mean) ** 2)

    class Function(object):

        @staticmethod
        def test_func(func, *range_):
            x = np.arange(*range_)
            y = func(x)
            plt.plot(x, y)
            plt.show()
            return

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
                                print '[BatchRun] %d left' % (len(run_list) + len(runnings) - 1, )
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
                                  % (len(x_columns), x_column, )
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

        class Functions(object):

            @staticmethod
            def get_variable_path(log_str):
                remove_path = [item['variable'] for item in RxModeling.LogAnalysis.single_re(
                    log_str, 'remove (.*),', ['variable'])]
                return remove_path

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

    class NpTools(object):

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
