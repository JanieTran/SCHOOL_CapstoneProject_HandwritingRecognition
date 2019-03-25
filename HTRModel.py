import tensorflow as tf
import utils
import sys
import numpy as np
import tensorflow.contrib.slim as slim

from tensorflow.python.training import moving_averages
from time import time
from tensorflow.python.ops.rnn import dynamic_rnn
from mdlstm import *


# Import constants from utils
FLAGS = utils.FLAGS
num_classes = utils.num_classes


class HTRModel(object):
    def __init__(self, mode):
        self.mode = mode

        # SparseTensor for ctc_loss
        self.labels = tf.sparse_placeholder(dtype=tf.int32)

        # Placeholder to store sequence length
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None])

        # Parameters
        self.learning_rate = FLAGS.initial_learning_rate
        self.decay_steps = FLAGS.decay_steps
        self.decay_rate = FLAGS.decay_rate
        self.batch_size = FLAGS.batch_size
        self.beta1 = FLAGS.beta1
        self.beta2 = FLAGS.beta2
        self.height = FLAGS.img_height
        self.width = FLAGS.img_width
        self.channels = FLAGS.img_channel
        self.hidden_size = FLAGS.num_hidden

        # Placeholder to store inputs
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.height, self.width, self.channels])
        self.is_training = True

    def build_graph(self):
        """
        TODO: what is this?
        :return:
        """
        self._build_model()
        self._build_train_op()

        self.merged_summary = tf.summary.merge_all()
