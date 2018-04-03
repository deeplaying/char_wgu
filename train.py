import tensorflow as tf
import numpy as np
from prepare_data import CharData
import time

# hyperparameters
num_hidden_units = 64
num_cells = 3
timesteps = 10
temperature = 0.9
no_epochs = 100
batch_size = 30
root_dir = "drive/app/char_rnn/"
data_path = root_dir+'gibber'

def network(inp, num_classes):

    inp = tf.unstack(inp, timesteps, axis=1)
    print(inp)

    cells = []
    for _ in range(3):
        cells.append(tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units))
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.static_rnn(lstm_cell, inp, dtype=tf.float32)

    weight = tf.Variable(tf.random_normal(shape=[num_hidden_units,num_classes]))
    bias = tf.Variable(tf.random_normal(shape=[num_classes]))
    prediction = tf.nn.sigmoid(tf.matmul(outputs[-1], weight) + bias, name="output_layer")
    return prediction

def main():

    data = CharData(data_path, batch_size, timesteps)
    num_classes = len(data.character_set)
    X = tf.placeholder(tf.float32, [None, timesteps, num_classes], name="input_data")
    Y = tf.placeholder(tf.float32, [None, num_classes], "expected_labels")
    graph = network(X, num_classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph, labels=Y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=4)
        for i in range(no_epochs):
            last_time = time.time()
            print(i)
            ret = True
            avg_cost = 0
            ctr = 0
            while ret:
                ret, feed_X, feed_Y = data.get_next_batch()
                print(ret)
                if ret:
                    ctr += 1
                    _cost, _ = sess.run((cost, train_step), feed_dict={X:feed_X, Y:feed_Y})
                    print(_cost)
                    avg_cost += _cost
            avg_cost = avg_cost / ctr
            print("epoch: {0}, cost: {1}; epoch took {2} seconds".format(i, avg_cost, time.time()-last_time))
            saver.save(sess, root_dir+'saved_weights/char_rnn', global_step=i)

if __name__ == "__main__":
    main()