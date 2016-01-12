import tensorflow as tf
import numpy as np
import math

# alternative activation function for a perceptron
SIGMOID = tf.sigmoid
TANH = tf.tanh
RELU = tf.nn.relu
SOFTPLUS = tf.nn.softplus

class Layer:
    def __init__(self, *params):
        self.size, self.layer_up_size, self.input, self.output, self.name, self.activation_func = params
        self.weights = None
        self.bias = None

class MLPModel:
    ''' Multi-Layer Perceptron Class

        (A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.
        Intermediate layers usually have as activation function tanh or the sigmoid 
        function while the top layer is a softmax layer or logistic regression layer
        class).
    '''
    # default activation function is sigmoid
    global activation_func
    activation_func = RELU

    # init by the input data palceholder with the size of each layer
    def __init__(self, x, layer_size=(3, 3, 3)):
        self.x = x
        self.layer_size = layer_size

    def set_activation_func(self, func):
        activation_func = func
    
    # set up neural network model 
    def inference(self):
        self.layers = self._define_layers()
        output = None
        for layer in self.layers:
            self._setup_one_layer(output, layer)
            output = layer.output
        return output
    
    def get_model(self, sess):
        return [ sess.run(layer.weights) for layer in self.layers ], [ sess.run(layer.weights) for layer in self.layers ]
   
    def get_l1_norm(self):
        return tf.add_n([tf.reduce_sum(layer.weights) for layer in self.layers])

    def get_l2_norm(self):
        return tf.add_n([tf.reduce_sum(tf.square(layer.weights)) for layer in self.layers])

    # define layer-wise params
    def _define_layers(self):
        ''' Define the structure of each layer in the network.
            The bottom layer accepts the input data.
            The top layer is a Logistic Regression layer(Can be changed to softmax if necessary)
        '''
        layers = list()
        for i, size in enumerate(self.layer_size):
            name = 'hidden' + str(i)
            if i == 0:
                layer = Layer(size, self.layer_size[1], self.x, None, name, activation_func)
            # Last layer is a Logistic Regression
            elif i == len(self.layer_size) - 1:
                layer = Layer(size, 1, None, None, name, SIGMOID)
            else:
                layer = Layer(size, self.layer_size[i+1], None, None, name, activation_func)
            layers.append(layer)
        return layers

    # set up one layer based on the params
    def _setup_one_layer(self, input, layer):
        ''' create one layer logic for ann model.

        Args:
          scope_name: the name of the current layer
          input: the data input to current layer
          layer_size: the number of the neurons of current layer
          layer_up_size: the number of the neurons of up layer       
        '''
        # stack the layer on the last layer we got
        if not layer.input:
            layer.input = input

	with tf.name_scope(layer.name):
            # set up activation logic, randomly initialize the weights and biases
            layer.weights = tf.Variable(tf.truncated_normal([layer.size, layer.layer_up_size], stddev= 0.3),name='weights')
            layer.bias = tf.Variable(tf.zeros([layer.layer_up_size]), name='biases')
            layer.output = layer.activation_func(tf.matmul(layer.input, layer.weights) + layer.bias)

def predict_accuracy(y, y_, sess):
    y = y.flatten().astype(int)
    y_ = y_.flatten().astype(int)
    print 1 - np.mean(np.abs(y - y_))

# Generate Samples
size = 1000
x_data = np.array((np.random.normal(0,1,size).astype("float32"), np.random.normal(0.5,0.5,size).astype("float32"), np.random.normal(0.5,0.5,size).astype("float32"))).T
w = np.array([[0.3, -0.1, 0]])
y_data = (1 / ( 1 + np.e ** -(x_data.dot(w.T) + 0.05)) > 0.5).astype(np.float32)
dimension = x_data.shape[1]
print x_data
print y_data

# Define the placeholder
x = tf.placeholder(tf.float32, [None, dimension])
y_ = tf.placeholder(tf.float32, [None, 1])

# Define the variable of the model
model = MLPModel(x, (dimension, 3, 3))
model.set_activation_func(SIGMOID)
y = model.inference()
y = tf.clip_by_value(y, 1e-5, 1-1e-5)

# Minimize the negative log likelihood.
lambda_ = 0.05
loss = (-tf.matmul(tf.transpose(y_), tf.log(y)) - tf.matmul(tf.transpose(1-y_), tf.log(1-y))) / size + 1/2 * lambda_ * model.get_l2_norm()
optimizer = tf.train.GradientDescentOptimizer(0.01)
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
        train_y = sess.run(y, {x:x_data}) > 0.5
        model_weights = model.get_model(sess)
        print(step, model_weights[0], model_weights[1], sess.run(loss, {x:x_data, y_:y_data}), sess.run(model.get_l2_norm()))
        predict_accuracy(train_y, y_data, sess)

print 'W:', model_weights[0]
print 'b:', model_weights[1]
