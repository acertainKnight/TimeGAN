import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# import os
# os.system('conda install xlrd')


def getData(path=r'FRED_MD_20200702.csv', var='UNRATE', d=None,
            ma=None, univariate=True):
    data = pd.read_csv(path, header=0, skiprows=[1])
    data['sasdate'] = pd.to_datetime(data['sasdate'], format="%m/%d/%y")
    future = data['sasdate'] > datetime(year=2050, month=1, day=1)
    data.loc[future, 'sasdate'] -= timedelta(days=365.25 * 100)
    data.set_index('sasdate', inplace=True)
    if univariate:
        data = pd.DataFrame(data[var])
    if d is not None:
        for i in d:
            _ = pd.DataFrame(data[var].diff(i))
            _.columns = ['d_{}'.format(i)]
            data = data.merge(_, how='left', right_index=True, left_index=True)
            data.columns = data.columns.str.rstrip('_x')
    if ma is not None:
        for i in ma:
            _ = pd.DataFrame(data[var].rolling(i).mean())
            _.columns = ['ma_{}'.format(i)]
            data = data.merge(_, how='left', right_index=True, left_index=True)
            data.columns = data.columns.str.rstrip('_x')
    if (d is not None) & (ma is not None):
        start = max(d + ma)
    else:
        if d is not None:
            start = max(d)
        elif ma is not None:
            start = max(ma)
        else:
            start = 0

    return data[start:-1]


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def get_val(x_train, y_train, index=450):
    x_ = x_train[index:]
    y_ = y_train[index:]
    x_train = x_train[:index]
    y_train = y_train[:index]
    x_val = x_train[::10]
    y_val = y_train[::10]
    x_train = x_train[np.mod(np.arange(x_train.shape[0]), 10) != 0]
    y_train = y_train[np.mod(np.arange(y_train.shape[0]), 10) != 0]
    x_val = np.append(x_val, x_[::2], axis=0)
    y_val = np.append(y_val, y_[::2], axis=0)
    x_train = np.append(x_train, x_[np.mod(np.arange(x_.shape[0]), 2) != 0], axis=0)
    y_train = np.append(y_train, y_[np.mod(np.arange(y_.shape[0]), 2) != 0], axis=0)
    return x_train, y_train, [x_val, y_val]


def scaleData(train, test, t):
    print(t)
    scalers = {}
    for i in range(train.shape[1]):
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        train[:, i, :] = scalers[i].fit_transform(train[:, i, :])

    for i in range(test.shape[1]):
        test[:, i, :] = scalers[i].transform(test[:, i, :])
    return train, test


def splitData(data, inval=24, outval=1, index=None, perc=0.8, var='univariate',
              single_step=True, val=True, val_index=None, holdout=False):
    values = data.values
    if index is None:
        index = int(values.shape[0] * perc)
    if val_index is None:
        val_index = index
    if var == 'univariate':
        x_train, y_train = univariate_data(values, 0,
                                           index, inval,
                                           outval)
        x_test, y_test = univariate_data(values, index-inval,
                                         None, inval,
                                         outval)

    else:
        col_idx = data.columns.get_loc(var)
        x_train, y_train = multivariate_data(values, values[:, col_idx], 0,
                                             index, inval,
                                             outval, 1,
                                             single_step=single_step)
        x_test, y_test = multivariate_data(values, values[:, col_idx], index-inval,
                                           None, inval,
                                           outval, 1,
                                           single_step=single_step)
    hold = None
    if holdout:
        x_hold, y_hold = x_test[int(len(x_test) / 2):], y_test[int(len(y_test) / 2):]
        x_test, y_test = x_test[:int(len(x_test) / 2)], y_test[:int(len(y_test) / 2)]
        _, x_hold = scaleData(np.concatenate((x_train.copy(), x_test.copy()), axis=0), x_hold, t='hold')
        hold = [np.array(x_hold), np.array(y_hold).reshape([-1, 1])]
    if val:
        x_train, y_train, val = get_val(np.array(x_train), np.array(y_train), val_index)
        _, x_val = scaleData(x_train.copy(), val[0], t='val')
        x_train, x_test = scaleData(x_train, x_test, t='test')
        val = (x_val, val[1])
        return np.array(x_train), np.array(y_train).reshape([-1, 1]), val, np.array(x_test), \
               np.array(y_test).reshape([-1, 1]), hold
    else:
        x_train, x_test = scaleData(x_train, x_test, t='test')
        return np.array(x_train), np.array(y_train).reshape([-1, 1]), None, np.array(x_test), \
               np.array(y_test).reshape([-1, 1]), hold


if __name__ == '__main__':
    data = getData()
    # print(len(data))
    x_train, y_train, val, x_test, y_test, hold = splitData(data, perc=.7, val=True, holdout=True)
    # print(len(y_train))
    # print(len(y_test))
    # print(len(hold[0]))
    #
    # print(len(y_train) + len(y_test))
    # plt.plot(data)
    # plt.show()
    # plt.plot(np.concatenate((y_train, y_test), axis=0))
    # plt.show()

    plt.plot(data.values[25:])
    plt.plot(np.concatenate((np.concatenate((y_train, y_test), axis=0), hold[1]), axis=0))
    plt.show()

    plt.plot(data.values[:-25])
    plt.plot(np.concatenate((np.concatenate((x_train, x_test), axis=0), hold[0]), axis=0)[:,0])
    plt.show()
    #
    # plt.plot(data.values[24:])
    # plt.plot(np.concatenate((y_train, np.concatenate((np.zeros((24,1)),y_test),axis=0)), axis=0))
    # plt.show()