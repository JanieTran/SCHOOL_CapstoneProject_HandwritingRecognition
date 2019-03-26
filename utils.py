import numpy as np
import tensorflow as tf


# tf.app.flags module is a functionality provided by TensorFlow
# to implement command line flags
flags = tf.app.flags

flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
flags.DEFINE_float('initial_learning_rate', 1e-3, 'initial learning rate')

flags.DEFINE_integer('img_height', 60, 'image height')
flags.DEFINE_integer('img_width', 180, 'image width')
flags.DEFINE_integer('img_channel', 1, 'image channels as input')

flags.DEFINE_integer('max_stepsize', 32, 'max stepsize in LSTM and output of last layer in CNN')
flags.DEFINE_integer('num_hidden', 50, 'number of hidden units in LSTM')
flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('save_steps', 1000, 'step to save checkpoint')
flags.DEFINE_integer('validation_steps', 1000, 'steps for validation')

flags.DEFINE_float('decay_rate', 0.98, 'learning rate decay rate')
flags.DEFINE_float('beta1', 0.9, 'parameter of Adam optimizer beta1')
flags.DEFINE_float('beta2', 0.999, 'Adam parameter beta2')

flags.DEFINE_integer('decay_steps', 10000, 'learning rate decay step')
flags.DEFINE_float('momentum', 0.9, 'momentum')

flags.DEFINE_string('train_dir', './imgs/train/', 'train data')
flags.DEFINE_string('val_dir', './imgs/val', 'validation data')
flags.DEFINE_string('infer_dir', './imgs/infer/', 'infer data')
flags.DEFINE_string('log_dir', './log', 'logging dir')
flags.DEFINE_string('mode', 'train', 'train, val, or infer')
flags.DEFINE_integer('num_gpus', 1, 'number of GPUS')

FLAGS = flags.FLAGS

# Number of class = 26 lowercase + 26 uppercase + blank + space
# num_classes = 26 + 1 + 1
num_classes = 3 + 2 + 10 + 1 + 1

max_print_len = 100

# Character set
# char_set = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_set = '0123456789+-*()'

encode_maps = {}
decode_maps = {}

for i, char in enumerate(char_set, start=1):
    encode_maps[char] = i
    decode_maps[i] = char

# Space character
SPACE_INDEX = 0
SPACE_TOKEN = ' '
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def calculate_accuracy(original_seq, decoded_seq, ignore_value=-1, is_print=False):
    """
    Calculate accuracy score of recognition
    :param original_seq: Original sequence
    :param decoded_seq: Decoded sequence
    :param ignore_value: Value to ignore when calculating accuracy
    :param is_print: Flag to control whether to print result
    :return: Accuracy score
    """
    # If 2 sequences are of different lengths, wrong recognition
    if len(original_seq) != len(decoded_seq):
        print('Original sequence is different from decoded sequence in length')
        return 0

    count = 0
    # For each character in original sequence
    for i, origin_label in enumerate(original_seq):
        # Get the corresponding character in decoded label
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]

        # If call for print
        if is_print and i < max_print_len:
            # Print out origin character and decoded character
            print('Sequence {}: origin: {} - decoded: {}'.format(i, origin_label, decoded_label))

            # Save both in test file
            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        # Increase count per correct decoding
        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """
    Create a sparse representation of x
    :param sequences: A list of lists of type dtype where
        each element is a sequence
    :param dtype: Data type of elements in sequences
    :return: A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    # For each sequence
    for n, seq in enumerate(sequences):
        # indices extend [{(n,0), (n, 1), (n,2), ... , (n, len(seq) - 1)}]
        indices.extend(zip([n] * len(seq), range(len(seq))))
        # values extend the sequence itself
        values.extend(seq)

    # Convert to NPArray
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(seq), np.asarray(indices).max(axis=0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def eval_expression(encoded_list):
    """
    TODO: what is this?
    :param encoded_list: TODO: what is this?
    :return:
    """
    eval_result = []

    for item in encoded_list:
        try:
            result = str(eval(item))
            eval_result.append(result)
        except:
            eval_result.append(item)
            continue

    with open(file='./result.txt') as f:
        for i in range(len(encoded_list)):
            f.write(encoded_list[i] + ' ' + eval_result[i] + '\n')

    return eval_result
