import tensorflow as tf
import numpy as np

class Classifier(tf.keras.Model):
    def __init__(self,n_layers = None, input_shape=None,neurons = None,length_output = None):

        super(Classifier,self).__init__()
        self.n = n_layers
        self.lay = {}
        self.first_layer = tf.keras.layers.Dense(neurons, activation = 'relu',kernel_initializer = tf.keras.initializers.he_uniform())
        for i in range(n_layers):
            self.lay[str(i)] = tf.keras.layers.Dense(neurons, activation = 'relu',kernel_initializer = tf.keras.initializers.he_uniform())
        self.last_layer = tf.keras.layers.Dense(length_output, activation = 'softmax',kernel_initializer = tf.keras.initializers.he_uniform())

    def call(self,inputs,training = False):
        x = self.first_layer(inputs)
        for i in range(self.n):
            x = self.lay[str(i)](x)
        x = self.last_layer(x)
        return x   