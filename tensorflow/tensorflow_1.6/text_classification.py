import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import re
import pandas as pd


# Load all files from a directory in a DataFrame
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            """re 模块检查一个字符串是否与某种模式匹配，\d 表示匹配数字， \w表示匹配字母或数字
                * 表示任意个字符， + 表示至少一个字符， ？ 表示0个或1个字符， {n}表示n个字符
                  {n, m}表示至少n个，至多m个字符
                  在使用re模块时，最好前面加r，这样可以避免转义情况"""
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
         fname="aclImdb.tar.gz",
         origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
         extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))
    print(os.path.dirname(dataset))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))
    return train_df, test_df


tf.logging.set_verbosity(tf.logging.ERROR)
train_df, test_df = download_and_load_datasets()
print(train_df.head())

train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)

predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)

predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=1000)
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.predict(input_fn=predict_test_input_fn)



print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))
