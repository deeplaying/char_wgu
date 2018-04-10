'''This module samples some data from a saved checkpoint'''
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

import argparse
import os
import tensorflow as tf
from prepare_data import CharData
import numpy as np
import pickle
from tqdm import tqdm


def sample(sample_type, prediction, temperature=0.9):
    
    if sampling_type==0:
        return np.argmax(prediction)
    else:
        sample_exp = np.exp(prediction) / temperature
        sample_reduce_mean = sample_exp / np.sum(sample_exp)
        prediction_real = np.random.choice(range(len(prediction)), 1, p=sample_reduce_mean)
        return prediction_real[0]


def get_meta_file_path(save_dir):
    '''Return the first .meta file in @param save_dit'''
    meta_file = ''
    for _f in os.listdir(save_dir):
        if _f[-5:] == '.meta':
            meta_file = os.path.join(save_dir, _f)
            break
    print(meta_file)
    return meta_file


def vectorize(text_to_vectorize, character_set):
    temp = np.zeros((1, len(text_to_vectorize), len(character_set)))
    for i, j in enumerate(text_to_vectorize):
        temp[0][i][character_set.index(j)] = 1
    return temp


def main():
    '''Run the script i guess'''
    #build the arguments parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save-dir', type=str, default='saved-checkpoints',
                        help='directory with the checkpoints to sample from')
    parser.add_argument('--n', type=int, default='100',
                        help='number of character to sample')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='timesteps to unravel the graph')
    parser.add_argument('--sampling-type', type=int, default=1,
                        help='sampling-type, 0-argmax, 1-exponential')
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='temperature for exponential sampling. between 0 & 1')    
    args = vars(parser.parse_args())

    save_dir = args['save_dir']
    sample_size = args['n']
    timesteps = args['timesteps']
    sampling_type = args['sampling_type']
    temperature = args['temperature']

    checkpoint_file = tf.train.latest_checkpoint(save_dir)
    meta_file = get_meta_file_path(save_dir)
    seed_file = os.path.join(save_dir, 'seed.txt')
    seed_data = CharData(seed_file, 1, 10)
    character_set = seed_data.character_set
    random_initialization = seed_data.random_seed(timesteps)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, checkpoint_file)
        input_ph = tf.get_default_graph().get_tensor_by_name('input_data:0')
        op_to_restore = tf.get_default_graph().get_tensor_by_name("output_layer:0")
        all_text = random_initialization[:]
        with tqdm(total=sample_size) as pb:
            for i in range(sample_size):
                text_input = vectorize(all_text[-timesteps:], character_set)
                out_vec = sess.run(op_to_restore, feed_dict={input_ph:text_input})[0]
                sampled_output = sample(sampling_type, out_vec, temperature=0.9)
                all_text = all_text + character_set[sampled_output]
                pb.update(i)
        print(all_text)


if __name__ == '__main__':
    main()
