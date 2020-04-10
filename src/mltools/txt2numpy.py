import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from keras.utils.np_utils import to_categorical

from keras.preprocessing import sequence


class TextConverter(object):
    def __init__(self, directory_or_fileiter, ycolumns, xcolumns,
                 shuffle=True, skip_header=0,
                 xtx=None, ytx=to_categorical, xmasks=None, num_samples=None, is_csv=True,
                 isTimeSeries=True):
        self.files = directory_or_fileiter
        self.ycolumns = ycolumns
        self.xcolumns = xcolumns
        self.shuffle = shuffle
        self.skip_header = skip_header
        self.xtx = xtx
        self.ytx = ytx
        self.xmasks = xmasks
        self.num_samples = num_samples
        self.is_csv = is_csv
        self.isTimeSeries = isTimeSeries

    def convert(self):
        result = [None] * len(self.files)
        if self.is_csv:
            for idx, file in enumerate(self.files):
                # important to specify dtype explicitly to make sure tf can consume it
                result[idx] = np.genfromtxt(file, dtype=np.float32, delimiter=',', skip_header=self.skip_header)
        else:
            # parquet
            if self.isTimeSeries:
                for idx, file in enumerate(self.files):
                    table = pq.read_table(file)
                    df = table.to_pandas()
                    # a bit hacky for now
                    df['label'] = df.apply(lambda row: 1.0 if row.TYPE == 'EBikeRide' else 0.0, axis=1)
                    # Note that 'features' may contains varying length
                    # feature_sequence = df['features'].to_numpy().tolist()
                    # feature_np = np.array(sequence.pad_sequences(feature_sequence, maxlen=10), np.float32)
                    # result.shape = (N, 2)
                    result[idx] = df[['label', 'features']].to_numpy()
            else:
                for idx, file in enumerate(self.files):
                    table = pq.read_table(file)
                    # important to specify dtype explicitly to make sure tf can consume it
                    np_array = table.to_pandas().to_numpy(dtype=np.float32)
                    result[idx] = np_array
        result = np.concatenate(result)
        if self.shuffle:
            idx = np.random.permutation(result.shape[0])
            result = result[idx, :]
        if self.num_samples and self.num_samples <= result.shape[0]:
            result = result[0:self.num_samples, :]
        X = self.make_x(result)
        y = self.make_y(result)
        if self.xtx:
            X = self.xtx(X)
        if self.ytx:
            y = self.ytx(y)
        return X, y

    # def make_x(self, np1):
    #     result = np1[:, self.xcolumns]
    #     if self.xmasks:
    #         temp = [None] * len(self.xmasks)
    #         for i, index in enumerate(self.xmasks):
    #             temp[i] = np1[:, index]
    #         result = np.transpose(temp)
    #     return result

    # hack
    def make_x(self, np1):
        # note that np1[:, self.xcolumns] has shape of [N,] because each row is taken as an np array object
        # use np.stack to cast it to 2d np array
        result = np.stack(np1[:, self.xcolumns])
        if self.xmasks:
            temp = [None] * len(self.xmasks)
            for i, index in enumerate(self.xmasks):
                temp[i] = result[:, index]
            result = np.transpose(temp)
        return result

    def make_y(self, np1):
        temp = np1[:, self.ycolumns].astype(np.float32)
        # temp is 1d array. convert to 2d 'vector'
        return np.reshape(temp, (temp.shape[0], 1))


class SimpleGenerator(object):
    def __init__(self, ycolumns, xcolumns, file_list=None,
                 positive_list=None, negative_list=None,
                 shuffle=True, skip_header=0,
                 xtx=None, ytx=to_categorical, xmasks=None, batch_size=32, is_csv=True):
        assert(file_list is not None or (positive_list is not None and negative_list is not None))
        if file_list is not None:
            self.converters = [TextConverter([file], ycolumns=ycolumns, xcolumns=xcolumns,
                                             shuffle=shuffle, skip_header=skip_header,
                                             xtx=xtx, ytx=ytx, xmasks=xmasks, is_csv=is_csv) for file in file_list]
        else:
            self.converters = [TextConverter([positive_file, negative_file], ycolumns=ycolumns, xcolumns=xcolumns,
                                             shuffle=shuffle, skip_header=skip_header,
                                             xtx=xtx, ytx=ytx, xmasks=xmasks, is_csv=is_csv) for positive_file,
                               negative_file in zip(positive_list, negative_list)]
        self.curr_file_idx = 0
        self.file_size = len(self.converters)
        self.batch_size = batch_size

    def gen(self):
        while True:
            if self.curr_file_idx >= self.file_size:
                # start from zero
                self.curr_file_idx = 0
            X, y = self.converters[self.curr_file_idx].convert()
            # now iterate through X and y
            index = 0
            size = X.shape[0]
            if self.batch_size > size:
                print("batch_size larger than numpy array provided")
                return
            while index + self.batch_size <= size:
                result = (X[index:index + self.batch_size, :], y[index:index + self.batch_size, :])
                index += self.batch_size
                yield result
            # jump to next file
            self.curr_file_idx += 1


def dummy_generator(X, y, batch_size):
    index = 0
    size = X.shape[0]
    if batch_size > size:
        print("batch_size larger than numpy array provided")
        return
    while True:
        if index + batch_size > size:
            # start from 0
            index = 0
        result = (X[index:index+batch_size, :], y[index:index+batch_size, :])
        index += batch_size
        yield result

