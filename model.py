import tensorflow as tf


def SimpleLSTMNetwork(inp, num_classes, timesteps, num_cells, num_hidden_units):
    inp = tf.unstack(inp, timesteps, axis=1)
    cells = []
    for _ in range(num_cells):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units))
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.static_rnn(lstm_cell, inp, dtype=tf.float32)

    weight = tf.Variable(tf.random_normal(shape=[num_hidden_units,num_classes]))
    bias = tf.Variable(tf.random_normal(shape=[num_classes]))
    prediction = tf.nn.relu(tf.matmul(outputs[-1], weight) + bias, name="output_layer")
    return prediction, states

