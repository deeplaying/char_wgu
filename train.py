'''
            o==+--
            |  |\ \
            |  | \ \    ____________________
            |   \ \ \   |                  |
            |    \ \ \  |  +------------+  |
            |     \ \ \ |  |     (__)   |  |
            |      \ \ \|  |     (oo)   |  |
            |       \ \ |  | o\  .\/.   |  |
            |        \ \|  | | \/    \  |  |
          /---\       \ |  +------------+  |
         /     \       \|                  |
         |     |        |                  |
         \     /        |                  |
          \---/         |                  |
                        |                  |
                     --------------------------
                    (                          )
                     --------------------------
'''

import tensorflow as tf
import numpy as np
from prepare_data import CharData
import os
import argparse
from tqdm import tqdm
from model import SimpleLSTMNetwork

# hyperparameters
num_hidden_units = 128
num_cells = 3
timesteps = 50


def maybe_save_seed_file(save_dir, character_set, seed_text):
    seed_file_path = os.path.join(save_dir, "seed.txt")
    print(seed_file_path)
    if not os.path.exists(seed_file_path):
        seed_text= seed_text + ''.join(character_set)
        with open(seed_file_path, 'w') as f:
            print('Saving seed file to ', seed_file_path)
            f.write(seed_text)


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-file', type=str, default='gibber',
                        help='data file containing the text to train the model on')
    parser.add_argument('--save-dir', type=str, default='saved-checkpoints',
                        help='directory to save the trained weights')
    parser.add_argument('--train-epochs', type=int, default=1000,
                        help='number of epochs to run the training')
    parser.add_argument('--save-every', type=int, default=1,
                        help='number of epochs to wait between saving the checkpoints')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='number of data points in a batch')
    parser.add_argument('--retrain', type=int, default=0,
                        help='0 to train fresh, 1 to retrain from save-dir')
    parser.add_argument('--save-to-drive', type=int, default=0,
                        help='save to drive or not, 1 if yes')
    parser.add_argument('--drive-path', type=str, default='',
                        help='path at which google drive is mounted')    
    parser.add_argument('--learning-rate', type=float, default=10e-3,
                        help='learning rate at which to update the network')    
    args = vars(parser.parse_args())
    print('Using the following args: ', args)
    save_dir = args['save_dir']
    data_file = args['data_file']
    no_epochs = args['train_epochs']
    save_every = args['save_every']
    batch_size = args['batch_size']
    _to_retrain = bool(args['retrain'])
    save_to_drive = bool(args['save_to_drive'])
    drive_path = args['drive_path']
    learning_rate = args['learning_rate']

    #build our model and set our placeholders and data
    data = CharData(data_file, batch_size, timesteps)
    num_classes = len(data.character_set)
    X = tf.placeholder(tf.float32, [None, timesteps, num_classes], name="input_data")
    Y = tf.placeholder(tf.float32, [None, num_classes], name="expected_labels")
    graph, states = SimpleLSTMNetwork(X, num_classes, timesteps, num_cells, num_hidden_units)
    print(graph)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=graph, labels=Y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=4)
        if _to_retrain:
            checkpoint_file = tf.train.latest_checkpoint(save_dir)
            saver.restore(sess, checkpoint_file)
            print("Training with weights from ", checkpoint_file)
        else:
            sess.run(tf.global_variables_initializer())
        for i in range(no_epochs):
            data.begin_new_epoch()            
            print('Epoch: {0}'.format(i))
            progress_bar = tqdm(total=data.length_of_text-(data.length_of_text%batch_size))
            avg_cost = 0
            ctr = 0
            while data._train_data_left:
                feed_X, feed_Y = data.get_next_batch()
                ctr += 1
                _cost, _ = sess.run((loss, train_step), feed_dict={X:feed_X, Y:feed_Y})
                avg_cost += _cost
                progress_bar.update(batch_size)
            avg_cost = avg_cost / ctr
            progress_bar.update(batch_size)
            progress_bar.close()
            print("Epoch: {0}, Cost: {1}".format(i, avg_cost))
            if i % save_every == 0:
                saver.save(sess, os.path.join(save_dir, 'checkpoint'), global_step=i)
                if save_to_drive:
                    saver.save(sess, os.path.join(drive_path, 'checkpoint'), global_step=i)
                print('Saved checkpoint')
                maybe_save_seed_file(save_dir, data.character_set, data.random_seed(1000))

if __name__ == "__main__":
    main()
