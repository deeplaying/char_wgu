import tensorflow as tf
import numpy as np
import glob


class ModelConfig:

    def __init__ (self):
        #set some sensible defaults
        self.cells = [64]
        self.num_classes = 26
        self.


def BuildModel(config, input_placeholder):
    inp = tf.unstack(inp, timesteps, axis=1)
    cells = []
    for _ in range(3):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units))
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.static_rnn(lstm_cell, inp, dtype=tf.float32)

    weight = tf.Variable(tf.random_normal(shape=[num_hidden_units,num_classes]))
    bias = tf.Variable(tf.random_normal(shape=[num_classes]))
    prediction = tf.nn.relu(tf.matmul(outputs[-1], weight) + bias, name="output_layer")
    return prediction