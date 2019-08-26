import tensorflow as tf
import numpy as np
import utils

from scipy.misc import imresize
from PIL import Image


class FinalModel:
    def __init__(self):
        with tf.gfile.GFile('./checkpoint/crnn/frozen_model_custom.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def, name='CRNN')

        self.x = self.graph.get_tensor_by_name('CRNN/inputs:0')
        self.y = self.graph.get_tensor_by_name('CRNN/CTCBeamSearchDecoder:1')
        self.seq_len = self.graph.get_tensor_by_name('CRNN/sequence_length:0')

    # -----------------------------------------------------------------------------------

    def recognise(self, sess, inputs):
        y_out = sess.run(self.y, feed_dict={
            self.x: inputs,
            self.seq_len: [127]
        })
        return utils.decode_result(y_out)

    # -----------------------------------------------------------------------------------

    @staticmethod
    def resize_input(image):
        image_arr = np.array(image)
        rows, cols = np.shape(image_arr)

        # If image width is bigger than defined width
        if cols > utils.IMG_WIDTH:
            cols = utils.IMG_WIDTH
            ratio = float(utils.IMG_WIDTH) / cols
            rows = int(utils.IMG_HEIGHT * ratio)
            inputs = imresize(image_arr, size=(rows, utils.IMG_WIDTH))

        # If image width <= defined width
        else:
            inputs = np.zeros(shape=(utils.IMG_HEIGHT, utils.IMG_WIDTH))
            ratio = float(utils.IMG_HEIGHT) / rows
            new_width = int(cols * ratio)
            image_arr_resized = imresize(image_arr, size=(utils.IMG_HEIGHT, new_width))
            inputs[:, 0:min(utils.IMG_WIDTH, new_width)] = image_arr_resized[:, 0:utils.IMG_WIDTH]

        inputs = np.array(inputs).astype(np.float32) / 255.

        inputs = inputs - inputs.min()
        inputs = inputs / inputs.max()
        inputs = np.swapaxes(inputs, axis1=0, axis2=1)
        inputs = np.reshape(inputs, [1, 512, 32, 1])

        return inputs

    # -----------------------------------------------------------------------------------

    def run(self, image):
        with tf.Session(graph=self.graph) as sess:
            inputs = self.resize_input(image)
            return self.recognise(sess, inputs)
