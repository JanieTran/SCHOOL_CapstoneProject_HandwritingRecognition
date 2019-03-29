import sys
import time
import datetime
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops.rnn import dynamic_rnn

import HTRModel
import utils
from DataIterator import DataIterator


# Helpers
logger = logging.getLogger('Training for HTR')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    """
    Train model
    :param train_dir: directory for train data
    :param val_dir: directory for validation data
    :param mode: train/val MODE
    :return:
    """
    # Initialise model
    model = HTRModel.HTRModel(mode=mode)
    model.build_graph()

    # Load data
    print('\n----------Loading data----------')
    train_feeder = DataIterator(data_dir=train_dir)
    print('Train size:', train_feeder.size)

    val_feeder = DataIterator(data_dir=val_dir)
    print('Validation size:', val_feeder.size)

    # Batch size
    num_train_samples = train_feeder.size
    num_train_batches_per_epoch = int(num_train_samples / utils.BATCH_SIZE)
    print('num_train_samples:', num_train_samples)
    print('batch_size:', utils.BATCH_SIZE)
    print('num_train_batches_per_epoch:', num_train_batches_per_epoch)

    num_val_samples = val_feeder.size
    num_val_batches_per_epoch = int(num_val_samples / utils.BATCH_SIZE)
    print('num_val_samples', num_val_samples)
    print('num_val_batches_per_epoch: {}\n'.format(num_val_batches_per_epoch))

    # Shuffle validation indices
    shuffle_index_val = np.random.permutation(num_val_samples)

    # Config
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        # Global variables initialiser
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(logdir=utils.LOG_DIR + 'train/', graph=sess.graph)

        # Restore checkpoints
        if utils.RESTORE:
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir=utils.CHECKPOINT_DIR)
            if checkpoint:
                saver.restore(sess=sess, save_path=checkpoint)
                print('-----Restore from checkpoint', checkpoint)

        # Training
        print('\n----------Begin Training----------')
        for current_epoch in range(utils.NUM_EPOCHS):
            # Shuffle train indices
            shuffle_index = np.random.permutation(num_train_samples)

            # Initialise/Reset cost
            train_cost = 0

            # Capture time that epoch begins
            start_time = time.time()
            batch_time = time.time()

            # Tracing part
            for current_batch in range(num_train_batches_per_epoch):
                # Log time every 100 batches
                if (current_batch + 1) % 100 == 0:
                    print('Batch {} - time: {}'.format(current_batch, time.time() - batch_time))

                # Capture time that batch begins
                batch_time = time.time()

                # Get batch from indices
                indices = [shuffle_index[i % num_train_samples] for i in
                           range(current_batch * utils.BATCH_SIZE, (current_batch + 1) * utils.BATCH_SIZE)]
                batch_inputs, batch_seq_len, batch_labels = train_feeder.input_index_generate_batch(index=indices)

                # Feed dict to model
                feed_dict = {
                    model.inputs: batch_inputs,
                    model.labels: batch_labels,
                    model.seq_len: batch_seq_len
                }
                model.is_training = True

                # Run training
                summary_str, batch_cost, step, _ = sess.run(
                    fetches=[model.merged_summary, model.cost, model.global_step, model.train_op],
                    feed_dict=feed_dict
                )
                print('Step {} - Batch_cost {}'.format(step, batch_cost))

                # Calculate cost
                delta_batch_cost = batch_cost * utils.BATCH_SIZE
                train_cost += delta_batch_cost
                train_writer.add_summary(summary=summary_str, global_step=step)

                # Save checkpoint
                if step % utils.SAVE_STEPS == 1:
                    # Make directory of not existing
                    if not os.path.isdir(utils.CHECKPOINT_DIR):
                        os.mkdir(utils.CHECKPOINT_DIR)
                    # Log info
                    logger.info('Save checkpoint of step', step)
                    # Save session
                    saver.save(sess=sess,
                               save_path=os.path.join(utils.CHECKPOINT_DIR, 'htr-model'),
                               global_step=step)

                # Validation
                if step % utils.VALIDATION_STEPS == 0:
                    # Initialise batch accuracy
                    acc_batch_total = 0
                    learning_rate = 0

                    for j in range(num_val_batches_per_epoch):
                        # Get batch from indices
                        val_indices = [shuffle_index_val[i % num_val_samples] for i in
                                       range(j * utils.BATCH_SIZE, (j + 1) * utils.BATCH_SIZE)]
                        val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch(val_indices)

                        # Validation feed dict
                        val_feed_dict = {
                            model.inputs: val_inputs,
                            model.labels: val_labels,
                            model.seq_len:val_seq_len
                        }
                        model.is_training = False

                        # Run validation
                        dense_decoded, learning_rate = sess.run(
                            fetches=[model.dense_decoded, model.learning_rate_decay],
                            feed_dict=val_feed_dict
                        )

                        # Print decoded result
                        original_labels = val_feeder.get_labels(indices=val_indices)
                        accuracy = utils.calculate_accuracy(original_seq=original_labels, decoded_seq=dense_decoded,
                                                            ignore_value=-1, is_print=True)
                        acc_batch_total += accuracy

                    # Average accuracy
                    accuracy = (acc_batch_total * utils.BATCH_SIZE) / num_val_samples
                    avg_train_cost = train_cost / ((current_batch + 1) * utils.BATCH_SIZE)

                    # Capture time epoch ends
                    now = datetime.datetime.now()
                    timestamp = '[{}/{} {}:{}:{}]'.format(now.day, now.month, now.hour, now.minute, now.second)
                    epoch_info = 'Epoch {}/{}:'.format(current_epoch + 1, utils.NUM_EPOCHS)
                    params_results = 'lr = {}, train_cost = {}, acc = {},'.format(learning_rate, avg_train_cost, accuracy)
                    time_elapsed = 'time_elapsed = {}'.format(time.time() - start_time)
                    print(timestamp, epoch_info, params_results, time_elapsed)


def main(_):
    """
    Main function
    """
    if utils.NUM_GPUS == 0:
        dev = '/cpu:0'
    elif utils.NUM_GPUS == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 GPU.')

    print('Device:', dev)
    with tf.device(dev):
        if utils.MODE == 'train':
            train(train_dir=utils.TRAIN_DIR, val_dir=utils.VAL_DIR, mode=utils.MODE)


if __name__ == '__main__':
    tf.logging.set_verbosity(v=tf.logging.INFO)
    tf.app.run()
