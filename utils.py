import numpy as np


# Whether to restore from the latest checkpoint
RESTORE = False
CHECKPOINT_DIR = './checkpoint/'
INITIAL_LEARNING_RATE = 1e-4

IMG_HEIGHT = 64
# IMG_WIDTH = 180
IMG_WIDTH = 512
IMG_CHANNEL = 1

# Max stepsize in LSTM and output of last layer in CNN
MAX_STEPSIZE = 128
NUM_HIDDEN = 50
NUM_EPOCHS = 2
BATCH_SIZE = 1
SAVE_STEPS = 10
VALIDATION_STEPS = 20

DECAY_RATE = 0.98
# Parameter of Adam optimizer BETA1
BETA1 = 0.9
# Adam parameter BETA2
BETA2 = 0.999

DECAY_STEPS = 10000
MOMENTUM = 0.9

TRAIN_DIR = './imgs/train/'
VAL_DIR = './imgs/small_val/'
INFER_DIR = './imgs/infer/'
LOG_DIR = './log/'
# train, val or infer
MODE = 'train'
NUM_GPUS = 1

# Character set
CHAR_SET = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# CHAR_SET = '0123456789+-*()'

MAX_PRINT_LEN = 100

ENCODE_MAPS = {}
DECODE_MAPS = {}

for i, char in enumerate(CHAR_SET, start=1):
    ENCODE_MAPS[char] = i
    DECODE_MAPS[i] = char

# Space character
SPACE_INDEX = 0
SPACE_TOKEN = ''
ENCODE_MAPS[SPACE_TOKEN] = SPACE_INDEX
DECODE_MAPS[SPACE_INDEX] = SPACE_TOKEN

# Number of class
NUM_CLASSES = len(ENCODE_MAPS) + 1
# NUM_CLASSES = 3 + 2 + 10 + 1 + 1


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

    original_text = [DECODE_MAPS[c] for c in original_seq[0]]
    decoded_text = [DECODE_MAPS[c] for c in decoded_seq[0]]
    print('original_text:', ''.join(original_text))
    print('decoded_text:', ''.join(decoded_text))

    count = 0
    # For each character in original sequence
    for i, origin_label in enumerate(original_seq):
        # Get the corresponding character in decoded label
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]

        # If call for print
        if is_print and i < MAX_PRINT_LEN:
            # Print out origin character and decoded character
            print('Sequence {}: origin: {} - decoded: {}'.format(i, origin_label, decoded_label))

            # Save both in test file
            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        # Increase count per correct decoding
        if origin_label == decoded_label:
            count += 1

    # print('correct:', count, end=' ')

    return count * 1.0 / len(original_seq)


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
