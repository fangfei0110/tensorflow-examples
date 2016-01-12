import tensorflow as tf
import numpy as np

def predict_accuracy(y, y_):
    y = y.flatten().astype(int)
    y_ = y_.flatten().astype(int)
    print "Accuracy: ", 1 - np.mean(np.abs(y - y_))

# Generate Samples
size = 1000
x_data = np.array((np.random.normal(0,1,size).astype("float32"), np.random.normal(0.5,0.5,size).astype("float32"))).T
w = np.array([[0.3, -0.1]])
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

# Define L1 or L2 regulization
regulization = tf.reduce_sum(tf.abs(W))
l2_regulization = tf.reduce_sum(tf.square(W))
lambda_ = 0.002

# Minimize the negative log likelihood.
loss = (-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y))) / size + lambda_ * l2_regulization
learning_rate = tf.train.exponential_decay(0.002, 40000, 4000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(40000):
    sess.run(train, {x:x_data, y_:y_data})
    if step % 200 == 0:
        train_W = sess.run(W)
        train_b = sess.run(b)
        train_y = sess.run(y, {x:x_data}) > 0.5
        print(step, train_W, train_b, sess.run(loss, {x:x_data, y_:y_data}))
        predict_accuracy(train_y, y_data)

print 'W:', train_W
print 'b:', train_b
