import tensorflow as tf
import numpy as np

def predict_accuracy(train_y, y_data, sess):
    y_ = y_data.flatten().astype(int)
    print 1 - np.mean(np.abs(train_y - y_))

# Generate Samples
size = 100
x_data = np.array((np.random.normal(0,1,size).astype("float32"), np.random.normal(0.5,0.5,size).astype("float32"))).T
w = np.array([[0.3, -0.1]])
y_data = (1 / ( 1 + np.e ** -(x_data.dot(w.T) + 0.05)) > 0.5).astype(np.float32)
print x_data
print y_data

# Define the variable of the model
dimension = x_data.shape[1]
W = tf.Variable(tf.random_uniform([1, dimension], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(x_data, tf.transpose(W)) + b)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Minimize the negative log likelihood.
loss = (-tf.matmul(tf.transpose(y_data), tf.log(y)) - tf.matmul(tf.transpose(1-y_data), tf.log(1-y))) / size
learning_rate = tf.train.exponential_decay(0.002, 40000, 4000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
# Fit the line.
for step in xrange(40000):
    sess.run(train)
    if step % 200 == 0:
        train_W = sess.run(W)
        train_b = sess.run(b)
        train_y = (sess.run(y).flatten() > 0.5).astype(int)
        print(step, train_W, train_b, sess.run(loss))
        predict_accuracy(train_y, y_data, sess)

print 'W:', train_W
print 'b:', train_b
