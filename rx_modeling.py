import numpy as np
import pandas as pd


class RxModeling(object):

    class DataPrepare(object):

        def __init__(self, data=None):
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

            self._data_train_mean = None
            self._data_train_std = None
            self._is_normalized = False

            self.x_columns = None
            self.y_column = None

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

        def set_data_from_origin_file(self, data_file=None, ask_or_bid='ask', y_length=60):
            if data_file is None:
                data_file = self.data_file
            assert isinstance(data_file, str) and data_file.endswith('.pic')
            data_dict = pd.read_pickle(data_file)
            data = data_dict['x']
            data_y = data_dict['y'][ask_or_bid].loc[data.index, y_length]
            data['y'] = np.sign(data_y) * np.log(np.abs(data_y) + 1)
            data.dropna(axis=0, inplace=True)
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

        def set_valid_range(self, valid_range=None):
            if valid_range is None:
                valid_range = self.valid_range
            assert self.data is not None
            self.valid_range = self._get_range(valid_range)

        def set_test_range(self, test_range=None):
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
                     ' ' * n_blanks + '%-*s' % (n_params, 'x_num') + str(len(list(self.x_columns))) + '\n' + \
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

        def set_train_test(self, output_type='DataFrame'):
            self.x_train = self.data[self.x_columns].iloc[self.train_range[0]:self.train_range[1]]
            self.y_train = self.data[self.y_column].iloc[self.train_range[0]:self.train_range[1]]
            self.x_valid = self.data[self.x_columns].iloc[self.valid_range[0]:self.valid_range[1]]
            self.y_valid = self.data[self.y_column].iloc[self.valid_range[0]:self.valid_range[1]]
            self.x_test = self.data[self.x_columns].iloc[self.test_range[0]:self.test_range[1]]
            self.y_test = self.data[self.y_column].iloc[self.test_range[0]:self.test_range[1]]

            self.x_train_sample = self.x_train.iloc[::self.train_sample_gap, :]
            self.y_train_sample = self.y_train.iloc[::self.train_sample_gap]
            self.x_valid_sample = self.x_valid.iloc[::self.train_sample_gap, :]
            self.y_valid_sample = self.y_valid.iloc[::self.train_sample_gap]

            if output_type in ('array', 'ndarray', 'Array'):
                self.x_train = self.x_train.values
                self.y_train = self.y_train.values
                self.x_valid = self.x_valid.values
                self.y_valid = self.y_valid.values
                self.x_test = self.x_test.values
                self.y_test = self.y_test.values

                self.x_train_sample = self.x_train_sample.values
                self.y_train_sample = self.y_train_sample.values
                self.x_valid_sample = self.x_valid_sample.values
                self.y_valid_sample = self.y_valid_sample.values
            elif output_type in ('DataFrame', 'dataFrame', 'dataframe'):
                pass
            else:
                raise Exception('Unknown type of output_type %s' % (output_type, ))

    class VariableSelection(object):
        pass

    class Model(object):
        pass

    