import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def layerNN(input, weights, bias):
    return tf.matmul(input, weights) + bias

class Ing_nn_model():
    def __init__(self, n_input, n_output, n_hidden_neurons):
        # Define input and target variables
        self.ingredients = tf.placeholder(tf.float32, [None, n_input])
        self.W = weight_variable([n_input, n_hidden_neurons])
        self.b = bias_variable([n_hidden_neurons])
        self.W2 = weight_variable([n_hidden_neurons, n_output])
        self.b2 = bias_variable([n_output])
        self.Layer = tf.nn.tanh(layerNN(self.ingredients, self.W, self.b))
        self.y = tf.nn.tanh(layerNN(self.Layer, self.W2, self.b2))



