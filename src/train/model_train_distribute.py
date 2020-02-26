# import argparse
# import horovod.keras as hvd
# import math
# import numpy as np
# import sys
# import tensorflow as tf
#
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint, History, TensorBoard, ReduceLROnPlateau
# from keras.optimizers import Adam
# from timeit import default_timer as timer
# from src.mltools.keras_tools.trainval_tensorboard import TrainValTensorBoard
# from src.train.utils import build_numpy_pn, build_generator_pn, fetch_file_list_pn, BATCH_SIZE
# from src.train.ar_model_train import TRAINING_DATA_DIR_N, TRAINING_DATA_DIR_P,\
#     VALIDATION_DATA_DIR_N, VALIDATION_DATA_DIR_P, MODEL_CHECKPOINT, LOG_DIR
# from src.train.ar_model_train import build_cnn_network, build_fc_network
#
# # add code path
# paths = ["/mnt/share/public/lucindaz/src"]
# for p in paths:
#     sys.path.append(p)
#
# # check https://github.com/uber/horovod/blob/master/examples/keras_mnist_advanced.py
# # for horovod config
# TRAIN_PORTION = 0.46
# VALIDATION_PORTION = 10
#
# LR = 5e-4
# NUM_SAMPLES = None
# EPOCHS = 30
# np.random.seed(12)
#
#
# # Horovod: initialize Horovod.
# hvd.init()
# # Horovod: pin GPU to be used to process local rank (one GPU per process)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = str(hvd.local_rank())
# K.set_session(tf.Session(config=config))
# # Horovod: adjust number of epochs based on number of GPUs.
# EPOCHS = int(math.ceil(EPOCHS * 1.0 / hvd.size()))
# # Scale the learning rate by number of workers when use distributed training.
# # Effective batch size in synchronous distributed training is scaled by the number of workers.
# # An increase in learning rate compensates for the increased batch size.
# LR = LR * hvd.size()
# NEW_BATCH_SIZE = BATCH_SIZE * hvd.size()
#
#
# def model_train(model_type='fc'):
#     start = timer()
#     if model_type == 'fc':
#         model_list = build_fc_network()
#     elif model_type == 'cnn':
#         model_list = build_cnn_network()
#     else:
#         print "Type not supported"
#         return
#     end = timer()
#     print("{} min taken for generating networks".format((end - start) / 60.0))
#
#     start = timer()
#     t_positive_list, negative_list = fetch_file_list_pn(
#         data_dir_p=TRAINING_DATA_DIR_P, data_dir_n=TRAINING_DATA_DIR_N, portion=TRAIN_PORTION)
#     tg = build_generator_pn(positive_list=t_positive_list, negative_list=negative_list, batch_size=NEW_BATCH_SIZE)
#     v_positive_list, v_negative_list = fetch_file_list_pn(
#         data_dir_p=VALIDATION_DATA_DIR_P, data_dir_n=VALIDATION_DATA_DIR_N, portion=VALIDATION_PORTION)
#     vg = build_numpy_pn(positive_list=v_positive_list, negative_list=v_negative_list, num_samples=NUM_SAMPLES)
#     end = timer()
#     print("{} min taken for generating input".format((end - start) / 60.0))
#
#     adam_wn = Adam(lr=LR)
#     # Horovod: add Horovod Distributed Optimizer.
#     opt = hvd.DistributedOptimizer(adam_wn)
#
#     for model_pair in model_list:
#         key = model_pair[0] + '_full_' + str(len(t_positive_list))
#         model = model_pair[1]
#         print(key)
#         print(model.summary())
#         model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
#
#         model_prefix = MODEL_CHECKPOINT + "_gps_imu_" + key + '_'
#
#         file_path = model_prefix + "{epoch:02d}-{val_acc:.2f}.h5"
#         checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=5)
#
#         callbacks_list = [
#             # Horovod: broadcast initial variable states from rank 0 to all other processes.
#             # This is necessary to ensure consistent initialization of all workers when
#             # training is started with random weights or restored from a checkpoint.
#             hvd.callbacks.BroadcastGlobalVariablesCallback(0),
#
#             # Horovod: average metrics among workers at the end of every epoch.
#             #
#             # Note: This callback must be in the list before the ReduceLROnPlateau,
#             # TensorBoard or other metrics-based callbacks.
#             hvd.callbacks.MetricAverageCallback(),
#
#             # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
#             # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
#             # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
#             hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
#
#             # Reduce the learning rate if training plateaues.
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=4, min_lr=1e-5),
#             TrainValTensorBoard(log_dir=LOG_DIR + '/' + key)
#
#         ]
#
#         # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#         if hvd.rank() == 0:
#             callbacks_list.append(checkpoint)
#
#         # Train the model.
#         # Horovod: the training will randomly sample 1 / N batches of training data and
#         # 3 / N batches of validation data on every worker, where N is the number of workers.
#         # Over-sampling of validation data helps to increase probability that every validation
#         # example will be evaluated.
#         model.fit_generator(generator=tg, steps_per_epoch=128*len(t_positive_list)//hvd.size(),
#                             epochs=EPOCHS, verbose=1, shuffle=True,
#                             use_multiprocessing=True, workers=10,
#                             callbacks=callbacks_list,
#                             max_queue_size=len(t_positive_list),
#                             validation_data=vg, validation_steps=128*len(v_positive_list)//hvd.size()*3)
#
#         model.save(model_prefix + '.h5')
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--type", help="cnn or fc")
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     args = parse_args()
#     model_train(model_type=args.type)
#
#
# if __name__ == "__main__":
#     main()
