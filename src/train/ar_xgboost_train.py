import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier, plot_importance

from src.train.utils import build_numpy, fetch_file_list


X_COLUMNS = slice(7, 85)
Y_COLUMNS = 0
np.random.seed(100)


TRAINING_DATA_DIR = './trainData/set1/'
VALIDATION_DATA_DIR = './validationData/set1/'

TRAIN_PORTION = 0.5
VALIDATION_PORTION = 0.5
MODEL_CHECKPOINT = '/Users/lucindazhao/strava/ml-local/snapshots/'
LOG_DIR = '/Users/lucindazhao/strava/ml-local/logs/'

XGBOOST_POSITIVE_WEIGHT = 10
IS_CSV = False


def xgboost_train():
    train_file_list = fetch_file_list(data_dir=TRAINING_DATA_DIR, portion=1)
    tg = build_numpy(file_list=train_file_list, num_samples=None, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None,
                     skip_header=1, shuffle=False, is_csv=IS_CSV)
    val_file_list = fetch_file_list(data_dir=VALIDATION_DATA_DIR, portion=1)
    vg = build_numpy(file_list=val_file_list, num_samples=None, xcolumns=X_COLUMNS, ycolumns=Y_COLUMNS, ytx=None,
                     skip_header=1, shuffle=False, is_csv=IS_CSV)

    x_train = copy.deepcopy(tg[0])
    y_train = copy.deepcopy(tg[1].reshape(-1))
    x_val = copy.deepcopy(vg[0])
    y_val = copy.deepcopy(vg[1].reshape(-1))
    del tg
    del vg

    count = np.sum(y_train)
    print("Number of Positive Training Windows: {}".format(count))
    print("Number of Negative Training Windows: {}".format(len(y_train) - count))

    eval_set = [(x_train, y_train), (x_val, y_val)]
    my_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_by_level=1,
                             colsample_bynode=1, colsample_bytree=0.8,
                             eta=0.03, gamma=0.1, learning_rate=0.1,
                             ax_delta_step=0, max_depth=6,
                             min_child_weight=3, missing=None,
                             n_estimators=600, n_jobs=1, nthread=None,
                             objective='binary:logistic', random_state=0,
                             reg_alpha=0, reg_lambda=1,
                             scale_pos_weight=XGBOOST_POSITIVE_WEIGHT,
                             seed=1234, subsample=0.8,
                             verbosity=2, tree_method='hist')
    my_model.get_xgb_params()
    # logloss here equivalent to CategoricalCrossEntropy in tensorflow
    trained = my_model.fit(x_train, y_train, early_stopping_rounds=15,
                     eval_metric=["logloss", "error"],
                     eval_set=eval_set, verbose=True)

    key = "xgboost-withClassWeight"
    file_path = MODEL_CHECKPOINT + key + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    trained.save_model(file_path)
    return trained


def xgb_cv(x_train, y_train):
    params = {'max_depth': 6, 'n_estimators': 1200, 'eta': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'min_child_weight': 3, 'gamma': 0.1, 'scale_pos_weight': 10, 'seed': 1234,
                         'verbosity': 2, 'tree_method': 'hist'}

    dtrain = xgb.DMatrix(x_train, label=y_train)

    cv_results = xgb.cv(
        params=params, dtrain=dtrain, num_boost_round=1200, nfold=10, stratified=False, folds=None,
        metrics=('logloss', 'error'), obj=None, feval=None, maximize=False, early_stopping_rounds=15,
        fpreproc=None, as_pandas=True, verbose_eval=2, show_stdv=True,
        seed=1234, callbacks=None, shuffle=False)
    return cv_results


def performance(model: XGBClassifier):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()

    print(model)
    plot_importance(model)
    plt.show()
