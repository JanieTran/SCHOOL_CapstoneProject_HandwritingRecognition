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
        self.momentum = FLAGS.momentum
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

    def _build_model(self):
        """
        Build model architecture
        """
        batch_norm_params = {
            'is_training': self.is_training,
            'decay': 0.9,
            'updates_collections': None
        }

        # Scoping mechanism: pass a set of shared arguments to
        #       each operation defined in the same scope
        with slim.arg_scope(list_ops_or_scope=[slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            # Inputs
            x = self.inputs
            x = tf.reshape(x, shape=[FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])

            # Block 1
            net = slim.conv2d(inputs=x, num_outputs=16, kernel_size=[3, 3], scope='conv1')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net, _ = mdlstm_while_loop(rnn_size=32, input_data=net,
                                       window_shape=[1, 1], dims=None, scope_n='mdlstm1')

            # Block 2
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], scope='conv2')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net, _ = mdlstm_while_loop(rnn_size=124, input_data=net,
                                       window_shape=[1, 1], dims=None, scope_n='mdlstm2')

        # Shapes
        ss = net.get_shape().as_list()
        shape = tf.shape(net)
        batch_size = shape[0]

        # Outputs
        outputs = tf.transpose(net, perm=[2, 0, 1, 3])
        outputs = tf.reshape(outputs, shape=[-1, shape[1] * shape[3]])

        # Train
        with tf.name_scope('Train'):
            # Params initialisation
            with tf.variable_scope('ctc_lost-1') as scope:
                # Initialiser
                trunc_norm = tf.truncated_normal_initializer(mean=0., stddev=0.75, seed=None, dtype=tf.float32)

                # Weights and bias
                W = tf.get_variable('w', shape=[ss[1] * ss[3], 200], initializer=trunc_norm)
                b = tf.get_variable('b', shape=[200], initializer=trunc_norm)

                # Weights and bias after dropout
                W1 = tf.get_variable('w1', shape=[200, num_classes], initializer=trunc_norm)
                b1 = tf.get_variable('b1', shape=[num_classes], initializer=trunc_norm)

            # Histograms
            tf.summary.histogram('histogram-w-ctc', W)
            tf.summary.histogram('histogram-b-ctc', b)

        # y_hat = W*x + b
        logits = tf.matmul(outputs, W) + b

        # Dropout
        logits = slim.dropout(logits, is_training=self.is_training, scope='dropout4')
        logits = tf.matmul(logits, W1) + b1

        # Transform outputs
        logits = tf.reshape(logits, shape=[batch_size, -1, num_classes])
        logits = tf.transpose(logits, perm=(1, 0, 2))

        self.logits = logits

    def _build_train_op(self):
        """

        """
        # Loss from CNN_LSTM_CTC
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar(name='cost', tensor=self.cost)
        tf.summary.histogram(name='cost', values=self.cost)

        # Learning rate and optimiser
        self.lr = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=self.decay_steps,
                                             decay_rate=self.decay_rate,
                                             staircase=True)
        self.optimiser = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                    momentum=self.momentum,
                                                    use_nesterov=True)
        self.optimiser = self.optimiser.minimize(loss=self.cost, global_step=self.global_step)

        # TODO: what is this?
        train_ops = [self.optimiser]
        self.train_op = tf.group(*train_ops)

        # Perform CTC beam search
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(inputs=self.logits,
                                                                    sequence_length=self.seq_len,
                                                                    merge_repeated=False)

        # Compute Levenshtein distance between sequences
        self.ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(self.decoded[0], dtype=tf.int32), truth=self.labels))
        tf.summary.scalar('ler', self.ler)

        # Fully-connected
        self.dense_decoded = tf.sparse_tensor_to_dense(sp_input=self.decoded[0], default_value=-1)
        tf.Print(input_=self.dense_decoded, data=[self.dense_decoded], message='dense_decoded')
        tf.Print(input_=tf.sparse_tensor_to_dense(sp_input=self.labels, default_value=-1),
                 data=[tf.sparse_tensor_to_dense(sp_input=self.labels, default_value=-1)],
                 message='self.labels')
