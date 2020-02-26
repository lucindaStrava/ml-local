from keras import backend as K
from keras.models import Model, load_model
from timeit import default_timer as timer
from src.train.utils import build_numpy_pn, build_generator_pn, fetch_file_list_pn, BATCH_SIZE

TEST_DATA_P = '/mnt/share/lucindaz/data_small/test_p/'
TEST_DATA_N = '/mnt/share/lucindaz/data_small/test_n/'
TEST_PORTION = 10


def model_test(model_path):
    start = timer()
    positive_list, negative_list = fetch_file_list_pn(data_dir_p=TEST_DATA_P, data_dir_n=TEST_DATA_N,
                                                      portion=TEST_PORTION)
    X, y = build_numpy_pn(positive_list=positive_list, negative_list=negative_list)
    end = timer()
    print("{} min taken for generating input".format((end - start) / 60.0))

    model = load_model(model_path)
    print(model.summary())
    print(model.metrics_names)
    print(K.eval(model.optimizer.lr))
    y_hat = model.predict(X)
    return y_hat, y


def main():
    output = model_test(model_path=None)


if __name__ == "__main__":
    main()
