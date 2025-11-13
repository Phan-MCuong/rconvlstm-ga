# models/convrnn_blocks.py
import tensorflow as tf
from tensorflow.keras import layers

class ConvRNN2DCell(layers.Layer):
    def __init__(self, filters, kernel_size, activation="tanh", use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.conv_x = None
        self.conv_h = None

    @property
    def state_size(self):
        # Keras RNN API cần state_size (H,W,C) nhưng H,W chỉ biết lúc build; dùng TensorShape(None)
        return [tf.TensorShape([None, None, self.filters])]

    def build(self, input_shape):
        # input_shape: (batch, H, W, C_in)
        c_in = int(input_shape[-1])
        self.conv_x = layers.Conv2D(
            self.filters, self.kernel_size, padding="same", use_bias=self.use_bias, name="conv_x")
        self.conv_h = layers.Conv2D(
            self.filters, self.kernel_size, padding="same", use_bias=self.use_bias, name="conv_h")
        super().build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]
        x_feat = self.conv_x(inputs)
        h_feat = self.conv_h(h_prev) if h_prev is not None else 0.0
        h = self.activation(x_feat + h_feat)
        return h, [h]

class ConvRNN2D(layers.Layer):
    def __init__(self, filters, kernel_size, return_sequences=False, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.cell = ConvRNN2DCell(filters, kernel_size, activation=activation)
        self.rnn = layers.RNN(self.cell, return_sequences=return_sequences)

    def call(self, x):
        # x: (B, T, H, W, C)
        # RNN mặc định kỳ vọng (B,T,...) rồi pass từng step
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        # states init = zeros
        H, W, C = x.shape[2], x.shape[3], self.cell.filters
        h0 = tf.zeros([B, H, W, C])
        return self.rnn(x, initial_state=[h0])
