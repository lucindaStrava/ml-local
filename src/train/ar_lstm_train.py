import argparse
import datetime
import numpy as np

import keras.optimizers.Adam as Adam
import keras.backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, History, ReduceLROnPlateau, TensorBoard
from keras.layers import Reshape, Dense, Dropout, Permute
from keras.models import load_model, Sequential
from timeit import default_timer as timer
from src.train.utils import build_numpy, build_generator, fetch_file_list

from keras.layers import recurrent

X_COLUMNS = 1
Y_COLUMNS = 0
BATCH_SIZE = 128
LR = 5e-4
# number of samples. A tiny number of samples for fast test
# For real training should reflect number of samples in whole training set
NUM_TRAIN_SAMPLES = BATCH_SIZE * 2
NUM_VALI_SAMPLES = BATCH_SIZE * 2

EPOCHS = 8
np.random.seed(100)

TRAINING_DATA_DIR = '/Users/lucindazhao/strava/ml-local/trainData/'
VALIDATION_DATA_DIR = '/Users/lucindazhao/strava/ml-local/validationData/'
TRAIN_PORTION = 1.1
VALIDATION_PORTION = 1.1
MODEL_CHECKPOINT = '/Users/lucindazhao/strava/ml-local/snapshots/v1/'
LOG_DIR = '/Users/lucindazhao/strava/ml-local/logs/'
IS_CSV = False


def build_lstm():
    # build CNN network
    # model definition
    modelGraph = Sequential()
    # make sure to define input layer
    input1 = Input(shape=(None,))
    # get N * 26 input matrix
    reshape2 = Reshape((26, -1), name="reshape2")

    # get 26 * N input matrix
    transpose = Permute((2, 1), name="transpose")
    layer1 = LSTM(units=100, return_sequences=True, name="lstm1")
    layer2 = LSTM(units=100, return_sequences=False, name="lstm2")
    dropout = Dropout(0.3, name="dropout")
    dense_layer = Dense(units=2, activation="softmax", kernel_regularizer=regularizers.l2(1e-3))
    modelGraph.add(input1)
    modelGraph.add(reshape2)
    modelGraph.add(transpose)
    modelGraph.add(layer1)
    modelGraph.add(layer2)
    modelGraph.add(dropout)
    modelGraph.add(dense_layer)

    print(modelGraph.summary())
    return 'lstm_' + str(LR), modelGraph


def model_train():
    """
    Use for fast iteration
    :param model_type:
    :return:
    """
    start = timer()
    model_pair = build_lstm()
    end = timer()
    print("{} min taken for generating networks".format((end - start) / 60.0))

    start = timer()
    train_file_list = fetch_file_list(data_dir=TRAINING_DATA_DIR, portion=TRAIN_PORTION)
    tg = build_generator(file_list=train_file_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)
    v_file_list = fetch_file_list(data_dir=VALIDATION_DATA_DIR, portion=VALIDATION_PORTION)
    vg = build_generator(file_list=v_file_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)
    end = timer()
    print("{} min taken for generating input".format((end - start) / 60.0))

    key = model_pair[0]
    model = model_pair[1]
    print(key)
    print(model.summary())

    history = History()
    log_dir = LOG_DIR + key + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = MODEL_CHECKPOINT + key + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_board = TensorBoard(log_dir, histogram_freq=5,
                                    write_grads=False, write_graph=False, profile_batch=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=4, min_lr=1e-5)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=5)

    callbacks_list = [history, tensor_board, reduce_lr, checkpoint]

    adam_wn = Adam(lr=LR)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam_wn, metrics=['accuracy'])

    model.fit_generator(generator=tg, steps_per_epoch=int(NUM_TRAIN_SAMPLES/BATCH_SIZE),
                        epochs=EPOCHS, verbose=1, shuffle=True,
                        use_multiprocessing=True, workers=10,
                        callbacks=callbacks_list,
                        max_queue_size=len(train_file_list),
                        validation_data=vg, validation_steps=int(NUM_VALI_SAMPLES/BATCH_SIZE),
                        class_weight={0: 0.1, 1: 1.0})

    model.save(file_path)
    return


def model_train_resume(model_path):
    """
    Resume training a saved model
    :param model_path:
    :return:
    """
    # load model
    model = load_model(model_path)

    start = timer()
    train_file_list = fetch_file_list(data_dir=TRAINING_DATA_DIR, portion=TRAIN_PORTION)
    tg = build_generator(file_list=train_file_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)
    v_file_list = fetch_file_list(data_dir=VALIDATION_DATA_DIR, portion=VALIDATION_PORTION)
    vg = build_generator(file_list=v_file_list, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)

    end = timer()
    print("{} min taken for generating input".format((end - start) / 60.0))
    print(model_path)
    print(model.summary())

    new_path = model_path + '_resume_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint = ModelCheckpoint(new_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=5)
    history = History()
    log_dir = LOG_DIR + model_path.split('/')[-1] \
              + '_resume_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensor_board = TensorBoard(log_dir, histogram_freq=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=1e-5)
    callbacks_list = [history, tensor_board, checkpoint, reduce_lr]

    # hack!!!
    print(K.eval(model.optimizer.lr))
    adam_wn = Adam(lr=LR)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam_wn, metrics=['accuracy'])
    print(K.eval(model.optimizer.lr))

    model.fit_generator(generator=tg, steps_per_epoch=int(NUM_TRAIN_SAMPLES/BATCH_SIZE),
                        epochs=EPOCHS, verbose=1, shuffle=True,
                        use_multiprocessing=True, workers=10,
                        callbacks=callbacks_list,
                        max_queue_size=len(train_file_list),
                        validation_data=vg, validation_steps=int(NUM_VALI_SAMPLES/BATCH_SIZE),
                        class_weight={0: 0.1, 1: 1.0})

    model.save(new_path)
    return


def model_train_small(model_type='fc'):
    """
    Use for fast iteration
    :param model_type:
    :return:
    """
    start = timer()
    model_pair = build_lstm()
    end = timer()
    print("{} min taken for generating networks".format((end - start) / 60.0))

    start = timer()
    train_file_list = fetch_file_list(data_dir=TRAINING_DATA_DIR, portion=1.1)
    tg = build_numpy(file_list=train_file_list, num_samples=None, xcolumns=X_COLUMNS,
                     ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)
    val_file_list = fetch_file_list(data_dir=VALIDATION_DATA_DIR, portion=1.1)
    vg = build_numpy(file_list=val_file_list, num_samples=None, xcolumns=X_COLUMNS,
                     ycolumns=Y_COLUMNS, ytx=None, is_csv=IS_CSV)
    end = timer()
    print("{} min taken for generating input".format((end - start) / 60.0))

    count = np.sum(tg[1], axis=0)
    print("Number of Positive Training Windows: {}".format(count[0]))
    print("Number of Negative Training Windows: {}".format(len(tg[1]) - count[0]))

    key = model_pair[0] + '_small'
    model = model_pair[1]
    print(key)
    print(model.summary())

    file_path = MODEL_CHECKPOINT + key + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    history = History()
    log_dir = LOG_DIR + key + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_board = TensorBoard(log_dir, histogram_freq=5,
                                       write_grads=False, write_graph=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=4, min_lr=1e-5)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=5)

    callbacks_list = [history, tensor_board, reduce_lr, checkpoint]

    adam_wn = Adam(lr=LR)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=adam_wn, metrics=['accuracy'])
    model.fit(x=tg[0], y=tg[1], validation_data=vg, batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1, shuffle=False,
              callbacks=callbacks_list, class_weight={0: 0.1, 1: 1.0})
    model.save(file_path)

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default='small', help="Train on a small dataset to observe overfitting. full or small")
    parser.add_argument("--type", help="cnn or fc", default='cnn')
    parser.add_argument("--path", default=None, help="model path")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to be used (on a single node)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.path:
        model_train_resume(model_path=args.path)
    else:
        if args.scale == 'small':
            model_train_small(model_type=args.type)
        elif args.scale == 'full':
            model_train(model_type=args.type, gpus=args.gpus)
        else:
            print ("Option not supported")


if __name__ == "__main__":
    main()
