# -*- coding:utf-8 -*-
# 迭代器的特点是：惰性求值（Lazy evaluation），即只有当迭代至某个值时，它才会被计算，
# 这个特点使得迭代器特别适合于遍历大文件或无限集合等，因为我们不用一次性将它们存储在内存中。
import itertools 

import pandas as pd
import tensorflow as tf


# Sets the threshold for what messages will be logged.
# 带有INFO级别的日志，每100步骤自动输出training-loss metrics到stderr。
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

def main(unused_argv):
    # skipinitialspace : Skip spaces after delimiter.
    # skiprows: 需要忽略的行数（从文件开始处算起），或需要跳过的行号列表（从0开始）。
    training_set = pd.read_csv("boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston_test.csv", skipinitialspace=True, skiprows=1, names= COLUMNS)
    prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10])
    regressor.train(input_fn=get_input_fn(training_set), steps=5000)
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))

    predictions = list(p["predictions"] for p in itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
    tf.app.run()
