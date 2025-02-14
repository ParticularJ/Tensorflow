import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

learning_rate = 0.01

scaler = StandardScaler()
housing = fetch_california_housing()
m, n = housing.data.shape
scaler_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaler_housing_data]

X = tf.placeholder(tf.float32, shape = (None, n+1), name = "X")
y = tf.placeholder(tf.float32, shape = (None, 1), name = "y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed = 42), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name = "mse")

print(error.op.name)
print(mse.op.name)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(mse)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 100
n_batches = int(np.ceil(m/100))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size = batch_index)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict = {X: X_batch,y: y_batch})            
                step  = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict= {X: X_batch, y: y_batch})

    print("best_theta %r" % theta.eval())
    
    save_path = saver.save(sess, "/tmp/data/my_model.ckpt")
file_writer.close()


















