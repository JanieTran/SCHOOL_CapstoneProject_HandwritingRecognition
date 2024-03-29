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
num_classes = utils.NUM_CLASSES


class HTRModel(object):
    def __init__(self, mode):
        self.mode = mode

        # SparseTensor for ctc_loss
        self.labels = tf.sparse_placeholder(dtype=tf.int32)

        # Placeholder to store sequence length
        self.seq_len = tf.placeholder(dtype=tf.int32, shape=[None])

        # Parameters
        self.learning_rate = utils.INITIAL_LEARNING_RATE
        self.decay_steps = utils.DECAY_STEPS
        self.decay_rate = utils.DECAY_RATE
        self.momentum = utils.MOMENTUM
        self.batch_size = utils.BATCH_SIZE
        self.beta1 = utils.BETA1
        self.beta2 = utils.BETA2
        self.height = utils.IMG_HEIGHT
        self.width = utils.IMG_WIDTH
        self.channels = utils.IMG_CHANNEL
        self.hidden_size = utils.NUM_HIDDEN

        # Placeholder to store inputs
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.height, self.width, self.channels])
        self.is_training = True

    def build_graph(self):
        """
        Build model
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

        print('\n----------Building model----------')
        # Scoping mechanism: pass a set of shared arguments to
        #       each operation defined in the same scope
        with slim.arg_scope(list_ops_or_scope=[slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            # Inputs
            x = self.inputs
            print('x:', x.get_shape().as_list())
            x = tf.reshape(x, shape=[utils.BATCH_SIZE, utils.IMG_HEIGHT, utils.IMG_WIDTH, utils.IMG_CHANNEL])
            print('x reshaped:', x.get_shape().as_list())

            # Block 1
            print('\n-----Block 1-----')
            net = slim.conv2d(inputs=x, num_outputs=16, kernel_size=[3, 3], scope='conv1')
            print('conv1:', net.get_shape().as_list())
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            print('pool1:', net.get_shape().as_list())
            net, _ = mdlstm_while_loop(rnn_size=32, input_data=net,
                                       window_shape=[1, 1], dims=None, scope_n='mdlstm1')
            print('mdlstm1:', net.get_shape().as_list())

            # Block 2
            print('\n-----Block 2-----')
            net = slim.conv2d(inputs=net, num_outputs=64, kernel_size=[3, 3], scope='conv2')
            print('conv2:', net.get_shape().as_list())
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            print('pool2:', net.get_shape().as_list())
            net, _ = mdlstm_while_loop(rnn_size=124, input_data=net,
                                       window_shape=[1, 1], dims=None, scope_n='mdlstm2')
            print('mdlstm2:', net.get_shape().as_list())

        # Shapes
        ss = net.get_shape().as_list()
        shape = tf.shape(net)
        batch_size = shape[0]

        # Outputs
        outputs = tf.transpose(net, perm=[2, 0, 1, 3])
        print('\noutputs transposed:', outputs.get_shape().as_list())
        outputs = tf.reshape(outputs, shape=[-1, shape[1] * shape[3]])
        print('outputs reshaped:', outputs.get_shape().as_list())

        # Train
        with tf.name_scope('Train'):
            # Params initialisation
            with tf.variable_scope('ctc_lost-1') as scope:
                # Initialiser
                trunc_norm = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)

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
        print('\nW: {} - b: {}'.format(W.get_shape().as_list(), b.get_shape().as_list()))
        print('logits:', logits.get_shape().as_list())

        # Dropout
        logits = slim.dropout(logits, is_training=self.is_training, scope='dropout4')
        print('logits dropout:', logits.get_shape().as_list())
        logits = tf.matmul(logits, W1) + b1
        print('W1: {} - b1: {}'.format(W1.get_shape().as_list(), b1.get_shape().as_list()))
        print('logits W1 b1:', logits.get_shape().as_list())

        # Transform outputs
        logits = tf.reshape(logits, shape=[batch_size, -1, num_classes])
        print('logits reshaped:', logits.get_shape().as_list())
        logits = tf.transpose(logits, perm=(1, 0, 2))
        print('logits transposed:', logits.get_shape().as_list())

        self.logits = logits

    def _build_train_op(self):
        """
        Training operations
        """
        print('\n----------Initialising training operations----------')
        # Global step
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        # Loss from CNN_LSTM_CTC
        print('logits:', self.logits.get_shape().as_list())
        self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar(name='cost', tensor=self.cost)
        tf.summary.histogram(name='cost', values=self.cost)

        # Learning rate and optimiser
        self.learning_rate_decay = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                              global_step=self.global_step,
                                                              decay_steps=self.decay_steps,
                                                              decay_rate=self.decay_rate,
                                                              staircase=True)
        self.optimiser = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_decay,
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
        self.ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(self.decoded[0], dtype=tf.int32),
                                                   truth=self.labels))
        tf.summary.scalar('ler', self.ler)

        # Fully-connected
        self.dense_decoded = tf.sparse_tensor_to_dense(sp_input=self.decoded[0], default_value=-1)

        # Printing
        tf.Print(input_=self.dense_decoded, data=[self.dense_decoded], message='dense_decoded')
        tf.Print(input_=tf.sparse_tensor_to_dense(sp_input=self.labels, default_value=-1),
                 data=[tf.sparse_tensor_to_dense(sp_input=self.labels, default_value=-1)],
                 message='self.labels')
