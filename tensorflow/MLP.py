import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("FLAG.data_dir/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

config = tf.contrib.learn.RunConfig(tf_random_seed = 42)
# 将X_train 转化成需要的数据类型
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)


dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = [300, 100], n_classes = 10, 
                                         feature_columns = feature_columns, config = config)

# 一个集成好的评估模型
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

dnn_clf.fit(x = X_train, y = y_train, batch_size = 50, steps = 10000)


from sklearn.metrics import accuracy_score

y_pred = dnn_clf.predict(X_test)
for i in y_pred:
    print("pred[%s]=" % i, y_pred[i])

a = accuracy_score(y_test, y_pred['classes'])
print("accuracy: ", a)

from sklearn.metrics import log_loss

y_pred_prob = y_pred['probabilities']
b = log_loss(y_test, y_pred_prob)
print("log_loss: ",b)


