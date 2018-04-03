import tensorflow as tf
import numpy as np
from prepare_data import CharData

root_dir = "drive/app/char_rnn/"
data_path = "char_wgu/gibber"
models_save_path = "./weights"

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(models_save_path + 'char_rnn-0.meta')
    saver.restore(sess, tf.train.latest_checkpoint(models_save_path))
    input_ph = tf.get_default_graph().get_tensor_by_name('input_data:0')
    op_to_restore = tf.get_default_graph().get_tensor_by_name("output_layer:0")
    print(sess.run(op_to_restore, feed_dict={input_ph:data.get_next_batch()[1]}))