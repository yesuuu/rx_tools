import sys
import os
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm


class RxModeling(object):

    class DataPrepare(object):

        class GetDataFunction(object):

            @staticmethod
            def get_data_from_file(data_file):
                assert isinstance(data_file, str)
                if data_file.endswith('.npy'):
                    data = pd.DataFrame(np.load(data_file))
                elif data_file.endswith('.csv'):
                    data = pd.read_csv(data_file, index_col=0, )
                else:
                    data = pd.read_pickle(data_file)
                return data

            @staticmethod
            def get_data_from_origin_file(data_file, ask_or_bid='ask', y_length=60):
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

            data_range = list(data_range[:2]) + [sample_gap]
            data_idx = self.data.index[data_numidx[::sample_gap]]

            return data_range, data_idx

        def normalize_data_using_train(self, norm_or_unnorm='norm'):
            """
            norm_or_unnorm in ('norm', 'unnorm')
            """
            if norm_or_unnorm == 'norm':
                assert not self._is_normalized
                data_train = self.data.loc[self.train_idx, :]
                self._data_train_mean = data_train.mean()
                self._data_train_std = data_train.std()
                self.data = (self.data - self._data_train_mean) / self._data_train_std
                self._is_normalized = True
            elif norm_or_unnorm == 'unnorm':
                assert self._is_normalized
                self.data = self.data * self._data_train_std + self._data_train_mean
                self._is_normalized = False
            else:
                raise Exception('Unknown norm_or_unnorm: %s' % (str(norm_or_unnorm), ))
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
        def __init__(self, file_format='log/%T', is_to_console=True):
            folder = os.path.split(file_format)[0]
            if not folder == '':
                if not os.path.isdir(folder):
                    os.makedirs(folder)
            if '%T' in file_format:
                time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
                file_format = file_format.replace('%T', time_str)
            elif '%D' in file_format:
                time_str = datetime.datetime.now().strftime('%Y-%m-%d')
                file_format = file_format.replace('%D', time_str)
            self.file_name = file_format
            self.is_to_console = is_to_console

        def start(self, is_print=False):
            self.log_obj = self.PrintLogObject(self.file_name, self.is_to_console)
            self.log_obj.start()
            if is_print:
                print '[log] log starts, to file %s' % (self.file_name, )

        def close(self):
            self.log_obj.close()

        def save(self, save_file, is_print=False):
            os.system('cp '+self.file_name+' '+save_file)
            if is_print:
                print '[log] log copy to %s' % (save_file, )

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