import tensorflow as tf
import numpy as np

# define the encoder
class Encoder(tf.keras.Model):
    def __init__(self, fillters = [8,16,32,64],kernel_sizes=[7,7,7,7],strides=[1,2,2,2],input_shape=None,num_of_layers = None,length_output = None):

        super(Encoder,self).__init__()
        
        assert(len(fillters)==num_of_layers)
        assert(len(kernel_sizes)==len(fillters))
        assert(len(strides)==len(fillters))
        activation = tf.nn.elu
        
        self.conv_layer_1 = tf.keras.layers.Conv2D(fillters[0],(kernel_sizes[0],kernel_sizes[0]),strides=(strides[0],strides[0]),activation = activation,padding='SAME',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_2 = tf.keras.layers.Conv2D(fillters[1],(kernel_sizes[1],kernel_sizes[1]),strides=(strides[1],strides[1]),activation = activation,padding='SAME',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_3 = tf.keras.layers.Conv2D(fillters[2],(kernel_sizes[2],kernel_sizes[2]),strides=(strides[2],strides[2]),activation = activation,padding='SAME',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_4 = tf.keras.layers.Conv2D(fillters[3],(kernel_sizes[3],kernel_sizes[3]),strides=(strides[3],strides[3]),activation = activation,padding='SAME',kernel_initializer = tf.keras.initializers.he_uniform())
        
        #self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_4 = tf.keras.layers.BatchNormalization()
        
        
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer_1 = tf.keras.layers.Dense(64,activation=activation,kernel_initializer=tf.keras.initializers.he_uniform())
        self.dense_layer_out = tf.keras.layers.Dense(length_output,activation=activation,kernel_initializer=tf.keras.initializers.he_uniform())

    
    def call(self, inputs, training = False):
        x = self.conv_layer_1(inputs)
        #x = self.batch_norm_1(x, training=training)
        x = self.conv_layer_2(x)
        #x = self.batch_norm_2(x, training=training)
        x = self.conv_layer_3(x)
        #x = self.batch_norm_3(x, training=training)
        x = self.conv_layer_4(x)
        #x = self.batch_norm_4(x, training=training)
        x = self.flatten_layer(x)
        x = self.dense_layer_1(x)
        x = self.dense_layer_out(x)
        return x
    
#define the DFNN
class DFNN(tf.keras.Model):
    def __init__(self,n_layers = None, input_shape=None,neurons = None,length_output = None):

        super(DFNN,self).__init__()
        activation = tf.nn.elu
        self.n = n_layers
        self.lay = {}
        self.first_layer = tf.keras.layers.Dense(neurons, activation = activation,kernel_initializer = tf.keras.initializers.he_uniform())
        for i in range(n_layers):
            self.lay[str(i)] = tf.keras.layers.Dense(neurons, activation = activation,kernel_initializer = tf.keras.initializers.he_uniform())
        self.last_layer = tf.keras.layers.Dense(length_output, activation = activation,kernel_initializer = tf.keras.initializers.he_uniform())

    def call(self,inputs,training = False):
        x = self.first_layer(inputs)
        for i in range(self.n):
            x = self.lay[str(i)](x)
        x = self.last_layer(x)
        return x    

#define the Decoder
class Decoder(tf.keras.Model):
    def __init__(self,fillters=[64,32,16,1],kernel_sizes = [7,7,7,7],strides = [2,2,2,1],input_kernel = None,num_of_layers = None, N = None):
        super(Decoder,self).__init__()

        assert(len(fillters)==num_of_layers)
        assert(len(kernel_sizes)==len(fillters))
        assert(len(strides)==len(fillters))

        activation = tf.nn.elu
        self.first_layer = tf.keras.layers.Dense(256, activation = activation,kernel_initializer = tf.keras.initializers.he_uniform())
        self.second_layer = tf.keras.layers.Dense(N, activation = activation,kernel_initializer = tf.keras.initializers.he_uniform())
        self.second_layer_reshaped = tf.keras.layers.Reshape((int(np.sqrt(N)/8),int(np.sqrt(N)/8),fillters[0]))
        self.conv_layer_1 = tf.keras.layers.Conv2DTranspose(fillters[0],(kernel_sizes[0],kernel_sizes[0]),strides=(strides[0],strides[0]),activation = activation,padding='same',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_2 = tf.keras.layers.Conv2DTranspose(fillters[1],(kernel_sizes[1],kernel_sizes[1]),strides=(strides[1],strides[1]),activation = activation,padding='same',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_3 = tf.keras.layers.Conv2DTranspose(fillters[2],(kernel_sizes[2],kernel_sizes[2]),strides=(strides[2],strides[2]),activation = activation,padding='same',kernel_initializer = tf.keras.initializers.he_uniform())
        self.conv_layer_4 = tf.keras.layers.Conv2DTranspose(fillters[3],(kernel_sizes[3],kernel_sizes[3]),strides=(strides[3],strides[3]),padding='same',kernel_initializer = tf.keras.initializers.he_uniform())
        self.out_1 = tf.keras.layers.Reshape((int(np.sqrt(N)),int(np.sqrt(N)),1))
        #self.out_2 = tf.keras.layers.Reshape((256,1))

        #self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        #self.batch_norm_4 = tf.keras.layers.BatchNormalization()

    def call(self,inputs,training = False):
        x = self.first_layer(inputs)
        #print(tf.shape(x))
        x = self.second_layer(x)
        #print(tf.shape(x))
        x = self.second_layer_reshaped(x)
        #print(tf.shape(x))
        #x = self.batch_norm_1(x, training = training)
        x = self.conv_layer_1(x)
        #print(tf.shape(x))
        #x = self.batch_norm_2(x, training = training)
        x = self.conv_layer_2(x)
        #print(tf.shape(x))
        #x = self.batch_norm_3(x, training = training)
        x = self.conv_layer_3(x)
        #print(tf.shape(x))
        #x = self.batch_norm_4(x, training = training)
        x = self.conv_layer_4(x)
        #print(tf.shape(x))
        x = self.out_1(x)
        #x = self.out_2(x)
        return x

