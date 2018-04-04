import tensorflow as tf
import numpy as np
from prepare_data import CharData

root_dir = "drive/app/char_rnn/"
data_path = "char_wgu/gibber"
models_save_path = "./weights/"

timesteps = 10

def sample(prediction, character_set, temperature=0.9):
    sample_exp = np.exp(prediction) / temperature
    sample_reduce_mean = sample_exp / np.sum(sample_exp)
    prediction_real = np.random.choice(character_set, 1, sample_reduce_mean)

def main():
    data = CharData(gibber)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(models_save_path + 'char_rnn-0.meta')
        saver.restore(sess, tf.train.latest_checkpoint(models_save_path))
        input_ph = tf.get_default_graph().get_tensor_by_name('input_data:0')
        output_layer = tf.get_default_graph().get_tensor_by_name("output_layer:0")

        random_seed = data.get_random_seed(timesteps)
        character_set = data.character_set
        for i in range(1000):
            input_vector = np.array([i for data.get_vector_for_id(len(character_set), character_set.index(i)) in random_seed[:-timesteps]])
            output_vector = sess.run(output_layer, feed_dict={input_ph:input_vector})
            _character = sample(output_vector, character_set)
            random_seed = random_seed + _character
        
        print(character_set)
if __name__ == '__main__':
    main()