import glob
import numpy as np
from keras.utils.np_utils import to_categorical
from src.mltools.txt2numpy import TextConverter, SimpleGenerator
from random import shuffle


np.random.seed(124)
BATCH_SIZE = 128
X_COLUMNS = slice(3, 39)
Y_COLUMNS = 2
COLORS = ['b', 'r', 'k', 'm', 'y', 'g']


# # Convert integer label to Category Classes
# def enc(s):
#     if s < 0.5:
#         return [1, 0]
#     return [0, 1]
#
# # call it when using for example categorical_crossentropy. When use binary_crossentropy, we can use label
#
# def ytx(y):
#     return np.apply_along_axis(enc, 1, y)


def generate_field_mask():
    """
    Return index to be included as features
    :return: a list of int
    """
    result = []
    for j in range(0, 9):
        for i in range(0, 25):
            result.append(30 * j + i)
    return result


def fetch_file_list_pn(data_dir_p, data_dir_n, portion):
    """
    data_dir_p for positive samples and data_dir_n for negative samples
    Fetch equal number of files from both folders.
    :param data_dir_p: str. should ends with '/'
    :param data_dir_n: str. should ends with '/'
    :param portion:
    :return:
    """
    # list of absolute file names for training
    positive_file_list = glob.glob(data_dir_p + "*.csv")
    negative_file_list = glob.glob(data_dir_n + "*.csv")

    # number of files to be used for training
    if portion > 1:
        use_files = min(len(positive_file_list), int(portion))
    else:
        use_files = max(int(len(positive_file_list) * portion), 1)
    print("Number of Files: {}".format(use_files * 2))

    return positive_file_list[:use_files], negative_file_list[:use_files]


def fetch_file_list(data_dir, portion):
    """
    Fetch files from both folder.
    :param data_dir: str. should ends with '/'
    :param portion:
    :return:
    """
    # list of absolute file names for training
    file_list = glob.glob(data_dir + "*.csv")
    if len(file_list) == 0:
        file_list = glob.glob(data_dir + "*.parquet")
    # number of files to be used for training
    if portion > 1:
        use_files = min(len(file_list), int(portion))
    else:
        use_files = max(int(len(file_list) * portion), 1)
    print("Number of Files: {}".format(use_files))
    return file_list[:use_files]


def build_generator_pn(positive_list, negative_list, batch_size=BATCH_SIZE,
                       xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, shuffle=True,
                       skip_header=1, xtx=None, ytx=to_categorical, xmasks=None, is_csv=True):
    """
    Use it for training on large amount of data
    :param data_dir:
    :param portion:
    :return:
    """
    # tg is the training data generator. len(tg) is the number of mini batch.
    # tg.__getitem__(0) for example is the first mini batch, where tg[0][0] is an array of features
    # and tg[0][1] is an array of labels
    # cgen = CSVFileGenerator(file_list, ycolumns=Y_COLUMNS,
    #                         xcolumns=X_COLUMNS, batch_size=batch_size,
    #                         handler='numpy', ytx=ytx, xmasks=None)
    # tg = cgen.train_gen()

    generator = SimpleGenerator(positive_list=positive_list,
                                negative_list=negative_list,
                                ycolumns=ycolumns,
                                xcolumns=xcolumns, shuffle=shuffle, skip_header=skip_header,
                                ytx=ytx, xtx=xtx, batch_size=batch_size, xmasks=xmasks, is_csv=is_csv)
    tg = generator.gen()
    return tg


def build_generator(file_list, batch_size=BATCH_SIZE, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, shuffle=True,
                    skip_header=1, xtx=None, ytx=to_categorical, xmasks=None, is_csv=True):
    """
    Use it for training on large amount of data
    :param data_dir:
    :param portion:
    :return:
    """
    generator = SimpleGenerator(file_list=file_list,
                                ycolumns=ycolumns,
                                xcolumns=xcolumns, shuffle=shuffle, skip_header=skip_header,
                                ytx=ytx, xtx=xtx, batch_size=batch_size, xmasks=xmasks, is_csv=is_csv)
    tg = generator.gen()
    return tg


def build_numpy_pn(positive_list, negative_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, shuffle=True, skip_header=1,
                 xtx=None, ytx=to_categorical, xmasks=None, num_samples=None, is_csv=True):

    file_list = positive_list + negative_list

    return build_numpy(file_list=file_list, xcolumns=xcolumns, ycolumns=ycolumns, shuffle=shuffle,
                       skip_header=skip_header, xtx=xtx, ytx=ytx, xmasks=xmasks, num_samples=num_samples,
                       is_csv=is_csv)


def build_numpy(file_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, shuffle=True, skip_header=1,
                 xtx=None, ytx=to_categorical, xmasks=None, num_samples=None, is_csv=True):

    converter = TextConverter(directory_or_fileiter=file_list, ycolumns=ycolumns, xcolumns=xcolumns,
                              shuffle=shuffle, skip_header=skip_header,
                              xtx=xtx, ytx=ytx, xmasks=xmasks, num_samples=num_samples, is_csv=is_csv)
    numpy_pair = converter.convert()

    return numpy_pair


def plot_histogram(histogram_2d_array, plt):
    n_groups = len(histogram_2d_array[0])
    x_axis = np.arange(n_groups)
    bar_width = 0.7 / len(histogram_2d_array)
    opacity = 0.8
    for i in range(0, len(histogram_2d_array)):
        plt.bar(x_axis + bar_width * i, histogram_2d_array[i], bar_width,
                alpha=opacity,
                color=COLORS[i % len(COLORS)],
                label=str(i))

    plt.title('Histogram')
    plt.xticks(x_axis + bar_width, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.legend()
    return
