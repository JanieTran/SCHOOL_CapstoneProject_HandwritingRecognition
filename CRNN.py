import os
import time
import tensorflow as tf
import utils

from tensorflow.contrib import rnn
from tensorflow.layers import conv2d as Conv2D
from tensorflow.layers import max_pooling2d as MaxPool2D


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def BiLSTM_K(inputs, sequence_length):
    """
        Bi-directional LSTM

    """
    # Mask for sequence length
    mask = tf.sequence_mask(lengths=sequence_length, dtype=tf.float32)
    mask = tf.expand_dims(input=mask, axis=-1)

    # BILSTM 1
    print('\n-----BiLSTM1-----')
    # LSTM cell
    lstm_1 = tf.keras.layers.LSTM(units=512, return_sequences=True)

    # Bi-directional LSTM cell
    bilstm_1 = tf.keras.layers.Bidirectional(lstm_1, merge_mode='concat')

    # Output
    bilstm_1_output = bilstm_1(inputs, mask=mask)
    print('bilstm_1:', bilstm_1_output.get_shape().as_list())

    # BILSTM 2
    print('-----BiLSTM2-----')
    # LSTM
    lstm_2 = tf.keras.layers.LSTM(units=512, return_sequences=True)

    # Bi-directional LSTM cell
    bilstm_2 = tf.keras.layers.Bidirectional(lstm_2, merge_mode='concat')

    # Output
    bilstm_2_output = bilstm_2(bilstm_1_output, mask=mask)
    print('bilstm_2:', bilstm_2_output.get_shape().as_list())
    print('-----Out of LSTM-----')

    return bilstm_2_output


def BiLSTM(inputs, sequence_length):
    """
        Bi-directional LSTM

    """
    # BiLSTM 1
    print('\n-----BiLSTM1-----')
    with tf.variable_scope(name_or_scope=None, default_name='bilstm-1'):
        # Forward
        lstm_fw_1 = rnn.BasicLSTMCell(num_units=256)
        # Backward
        lstm_bw_1 = rnn.BasicLSTMCell(num_units=256)

        inter_output, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw_1,
            cell_bw=lstm_bw_1,
            inputs= inputs,
            sequence_length=sequence_length,
            dtype=tf.float32
        )
        inter_output = tf.concat(inter_output, axis=2)
        print('bilstm_1:', inter_output.get_shape().as_list())

    # BiLSTM 2
    print('-----BiLSTM2-----')
    with tf.variable_scope(name_or_scope=None, default_name='bilstm-2'):
        # Forward
        lstm_fw_2 = rnn.BasicLSTMCell(num_units=256)
        # Backward
        lstm_bw_2 = rnn.BasicLSTMCell(num_units=256)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_fw_2,
            cell_bw=lstm_bw_2,
            inputs= inter_output,
            sequence_length=sequence_length,
            dtype=tf.float32
        )
        outputs = tf.concat(outputs, axis=2)
        print('bilstm_2:', outputs.get_shape().as_list())
        print('-----Out of LSTM-----')

    return outputs


def CNN(inputs):
    """
    Convolutional Neural Network

    """
    print('\n-----CNN-----')
    # Layer 1
    conv1 = Conv2D(inputs=inputs, filters=64, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv1:', conv1.get_shape().as_list())
    pool1 = MaxPool2D(inputs=conv1, pool_size=[2, 2], strides=2)
    print('Pool1:', pool1.get_shape().as_list())

    # Layer 2
    conv2 = Conv2D(inputs=pool1, filters=128, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv2:', conv2.get_shape().as_list())
    pool2 = MaxPool2D(inputs=conv2, pool_size=[2, 2], strides=2)
    print('Pool2:', pool2.get_shape().as_list())

    # Layer 3
    conv3 = Conv2D(inputs=pool2, filters=256, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv3:', conv3.get_shape().as_list())

    # Batch normalisation
    bnorm1 = tf.layers.batch_normalization(inputs=conv3)
    print('Bnorm1:', bnorm1.get_shape().as_list())

    # Layer 4
    conv4 = Conv2D(inputs=bnorm1, filters=256, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv4:', conv4.get_shape().as_list())
    pool4 = MaxPool2D(inputs=conv4, pool_size=[2, 2],
                      strides=[1, 2], padding='same')
    print('Pool4:', pool4.get_shape().as_list())

    # Layer 5
    conv5 = Conv2D(inputs=pool4, filters=512, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv5:', conv5.get_shape().as_list())

    # Batch normalisation
    bnorm2 = tf.layers.batch_normalization(inputs=conv5)
    print('Bnorm2:', bnorm2.get_shape().as_list())

    # Layer 6
    conv6 = Conv2D(inputs=bnorm2, filters=512, kernel_size=(3, 3),
                   padding='same', activation=tf.nn.relu)
    print('Conv6:', conv6.get_shape().as_list())
    pool6 = MaxPool2D(inputs=conv6, pool_size=[2, 2],
                      strides=[1, 2], padding='same')
    print('Pool6:', pool6.get_shape().as_list())

    # Layer 7
    conv7 = Conv2D(inputs = pool6, filters=512, kernel_size=(2, 2),
                   padding='valid', activation=tf.nn.relu)
    print('Conv7:', conv7.get_shape().as_list())
    print('-----Out of CNN-----')

    return conv7


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


class CRNN(object):
    def __init__(self):
        self.global_step = tf.Variable(initial_value=0, trainable=False)

    def build_model(self, batch_size):
        print('\n----------Building model----------')
        # Inputs
        inputs = tf.placeholder(dtype=tf.float32, name='inputs',
                                shape=[batch_size, utils.IMG_WIDTH, utils.IMG_HEIGHT, utils.IMG_CHANNEL])
        print('inputs:', inputs.get_shape().as_list())
        # Labels output
        labels = tf.sparse_placeholder(dtype=tf.int32, name='labels')

        # Sequence length
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # CNN
        cnn_output = CNN(inputs=inputs)
        cnn_output = tf.reshape(cnn_output, shape=[batch_size, -1, 512])
        print('cnn_output reshaped:', cnn_output.get_shape().as_list())

        # Max character count
        max_char_count = cnn_output.get_shape().as_list()[1]
        print('max_char_count:', max_char_count)

        # BiLSTM
        bilstm_output = BiLSTM(inputs=cnn_output, sequence_length=sequence_length)
        print('bilstm_output:', bilstm_output.get_shape().as_list())

        # Weights and bias
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[512, utils.NUM_CLASSES], stddev=0.1),
                        name='W')
        print('W:', W.get_shape().as_list())
        b = tf.Variable(initial_value=tf.constant(0., shape=[utils.NUM_CLASSES]),
                        name='b')
        print('b:', b.get_shape().as_list())

        # Logits
        logits = tf.reshape(bilstm_output, shape=[-1, 512])
        print('logits:', logits.get_shape().as_list())
        logits = tf.matmul(logits, W) + b
        print('logits linear:', logits.get_shape().as_list())
        logits = tf.reshape(logits, shape=[batch_size, -1, utils.NUM_CLASSES])
        print('logits reshaped:', logits.get_shape().as_list())

        # Final layer
        logits = tf.transpose(logits, perm=(1, 0, 2))
        print('logits transposed:', logits.get_shape().as_list())

        # Loss and cost
        loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=sequence_length)
        cost = tf.reduce_mean(loss)
        tf.summary.scalar(name='cost', tensor=cost)
        tf.summary.histogram(name='cost', values=cost)

        # Optimiser
        optimiser = tf.train.AdamOptimizer(learning_rate=utils.INITIAL_LEARNING_RATE)
        optimiser = optimiser.minimize(loss=cost, global_step=self.global_step)

        # Decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=logits,
            sequence_length=sequence_length,
            merge_repeated=True
        )
        dense_decoded = tf.sparse_tensor_to_dense(sp_input=decoded[0], default_value=-1)

        # Error rate
        edit_distance = tf.edit_distance(hypothesis=tf.cast(decoded[0], dtype=tf.int32), truth=labels)
        acc = tf.reduce_mean(edit_distance)
        tf.summary.scalar(name='distance', tensor=acc)
        tf.summary.histogram(name='distance', values=acc)

        return inputs, labels, sequence_length, logits, dense_decoded, \
               optimiser, acc, cost, max_char_count

    def build_graph(self, batch_size):
        (
            self.inputs, self.labels, self.sequence_length,
            self.logits, self.decoded, self.optimiser,
            self.acc, self.cost, self.max_char_count
        ) = self.build_model(batch_size=batch_size)

        self.merged_summary = tf.summary.merge_all()



