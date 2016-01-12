import tensorflow as tf
import numpy as np

def predict_accuracy(y, y_):
    y = y.flatten().astype(int)
    y_ = y_.flatten().astype(int)
    print "Accuracy: ", 1 - np.mean(np.abs(y - y_))

# Generate Samples
size = 40000
x_data = np.array((np.random.normal(0.8,0.2,size).astype("float32"), np.random.normal(0.5,0.5,size).astype("float32"), np.random.normal(0,0.5,size).astype("float32"))).T
# set the third dimension feature as "unimportant" feature to check if FTRL will "eliminate" this feature's weight
w = np.array([[0.5, -0.6, 0.01]])
y_data = (1 / ( 1 + np.e ** -(x_data.dot(w.T) + 0.05)) > 0.5).astype(np.float32)
dimension = x_data.shape[1]
print x_data
print y_data

# Define the placeholder
x = tf.placeholder("float", [None, dimension])
y_ = tf.placeholder("float", [None, 1])

# Define the variable of the model
W = tf.Variable(tf.random_uniform([1, dimension], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(x, tf.transpose(W)) + b)
# clipping y to avoid log(y) become infinite
y = tf.clip_by_value(y, 1e-10, 1-1e-10)

# Minimize the negative log likelihood.
loss = (-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y)))
optimizer = tf.train.FtrlOptimizer(0.03, l1_regularization_strength=0.01, l2_regularization_strength=0.01)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
# feed the sample with batch = 1
for sample_index in xrange(x_data.shape[0]):
    sess.run(train, {x:x_data[sample_index:sample_index+1, :], y_:y_data[sample_index:sample_index+1, :]})
    train_W = sess.run(W)
    train_b = sess.run(b)
    if sample_index % 200 == 0:
        print(sample_index, train_W, train_b, sess.run(loss / size, {x:x_data, y_:y_data}))

# End print the model and the training accuracy
print 'W:', train_W
print 'b:', train_b

train_y = sess.run(y, {x:x_data}) > 0.5
predict_accuracy(train_y, y_data)
