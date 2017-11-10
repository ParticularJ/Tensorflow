import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype = tf.float64)
    b = tf.get_variable("b", [1], dtype = tf.float64)
    y = W*features['x'] + b
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step,1))
    # EsimatorSpec connects subgraphs we built to the appropriate functionality
    return tf.estimator.EstimatorSpec(
        mode = mode, 
        predictions = y,
        loss = loss,
        train_op = train)

estimator = tf.estimator.Estimator(model_fn = model_fn)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5. , 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size = 4, num_epochs = None, shuffle = True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size = 4, num_epochs = 1000, shuffle = False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size = 4, num_epochs = 1000, shuffle = False)

# Train
estimator.train(input_fn = input_fn, steps = 1000)

train_metrics = estimator.evaluate(input_fn = train_input_fn)
eval_metrics = estimator.evaluate(input_fn = eval_input_fn)
print("train metric: %r" % train_metrics)
print("eval metric: %r" % eval_metrics)
