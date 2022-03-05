import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputLayer, Conv2D, Lambda
from tensorflow import keras


class Policy(tf.keras.Model):

    def __init__(self, action_space, network_size, depth=10):
        #: PolicyとValueがネットワークを共有するアーキテクチャ
        super(Policy, self).__init__()

        size = network_size
        self.denses = []
        for i in range(depth-1):
            self.denses.append(Cx2DAffine(size, activation="normalize"))

        self.dense3 = Cx2DAffine(size//2, activation="normalize")
        self.dense4 = kl.Dense(size, activation="relu")
        self.logits = kl.Dense(action_space)

    @tf.function
    def call(self, x):

        for layer in self.denses:
            x = layer(x)

        x = self.dense3(x)

        #: reshape complex to real
        x_real = x[:, 0, :]
        x_imag = x[:, 1, :]
        x = keras.layers.concatenate([x_real, x_imag], axis=1)

        x1 = self.dense4(x)
        logits = self.logits(x1)
        action_probs = tf.nn.softmax(logits)

        return action_probs

    def predict(self, states):

        states = np.atleast_2d(states)

        action_probs = self(states)

        return action_probs


class Cx2DAffine(Layer):

    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(Cx2DAffine, self).__init__(**kwargs)

    def build(self, input_shape):
        #: initialization
        self.weight_real = self.add_weight(
            name="weight_real",
            shape=(input_shape[2], self.output_dim),
            initializer="glorot_uniform"
        )

        self.weight_imag = self.add_weight(
            name="weight_imag",
            shape=(input_shape[2], self.output_dim),
            initializer="glorot_uniform"
        )

        self.bias_real = self.add_weight(
            name="bias_real",
            shape=(1, self.output_dim),
            initializer="zeros"
        )

        self.bias_imag = self.add_weight(
            name="bias_imag",
            shape=(1, self.output_dim),
            initializer="zeros"
        )

        super(Cx2DAffine, self).build(input_shape)

    def call(self, x):
        #: input
        x_real = Lambda(lambda x: x[:, 0, :], output_shape=(x.shape[2], ))(x)
        x_imag = Lambda(lambda x: x[:, 1, :], output_shape=(x.shape[2], ))(x)

        #: computation according to mpx
        real = K.dot(x_real, self.weight_real) - K.dot(x_imag, self.weight_imag)
        imag = K.dot(x_real, self.weight_imag) + K.dot(x_imag, self.weight_real)

        real = real + self.bias_real
        imag = imag + self.bias_imag

        #: activation
        if self.activation == "normalize":
            length = K.sqrt(K.pow(real, 2) + K.pow(imag, 2))
            real = real / length
            imag = imag / length

        # expand for concatenation
        real = K.expand_dims(real, 1)
        imag = K.expand_dims(imag, 1)

        # merge mpx
        cmpx = keras.layers.concatenate([real, imag], axis=1)
        return cmpx

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2, self.output_dim)