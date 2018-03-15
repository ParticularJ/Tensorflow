import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    # header=0，表示index从0开始
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, test_path


def load_data(y_name="Species"):
    train_path, test_path = maybe_download()
    # header=0.ignore the first row of the CSV file.
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    # pop表示弹出对应列
    train_x, train_y = train, train.pop(y_name)
    a=dict(train_x)
    

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # repeat（）无参数，无限次重复
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size not be none"
    dataset = dataset.batch(batch_size)

    return dataset
