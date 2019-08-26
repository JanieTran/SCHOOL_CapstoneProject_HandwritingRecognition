import os
import tensorflow as tf
import utils

from CRNN import CRNN


ckpt_state = tf.train.get_checkpoint_state(utils.CRNN_CHECKPOINT_DIR)
ckpt = ckpt_state.model_checkpoint_path

model = CRNN()
model.build_graph(batch_size=utils.BATCH_SIZE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=utils.CRNN_CHECKPOINT_DIR)
    saver.restore(sess, checkpoint)

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=['CTCBeamSearchDecoder']
    )

    with tf.gfile.GFile('./checkpoint/crnn/frozen_model_custom.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
