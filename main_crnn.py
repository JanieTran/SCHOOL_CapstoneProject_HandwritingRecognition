import sys
import time
import datetime
import os

import numpy as np
import tensorflow as tf

import utils
from CRNN import CRNN
from DataIterator import DataIterator
from DataManager import DataManager


def train():
    """
    Train model

    """

    # Load data
    print('\n----------Loading data----------')
    train_feeder = DataManager(train=True)
    print('Train size:', train_feeder.size)

    # Batch size
    num_train_samples = train_feeder.size
    num_train_batches_per_epoch = int(num_train_samples / utils.BATCH_SIZE)
    print('num_train_samples:', num_train_samples)
    print('batch_size:', utils.BATCH_SIZE)
    print('num_train_batches_per_epoch:', num_train_batches_per_epoch)

    # Initialise model
    model = CRNN()
    model.build_graph(batch_size=utils.BATCH_SIZE)

    # Config
    print()
    config = tf.ConfigProto(allow_soft_placement=True)
    log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = os.path.join(utils.CRNN_LOG_DIR, log_timestamp) + '/'

    if not os.path.isdir(utils.CRNN_LOG_DIR):
        os.mkdir(utils.CRNN_LOG_DIR)

    with tf.Session(config=config) as sess:
        # Global variables initialiser
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

        # Restore checkpoints
        if utils.RESTORE:
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir=utils.CRNN_CHECKPOINT_DIR)
            if checkpoint:
                saver.restore(sess=sess, save_path=checkpoint)
                print('-----Restore from checkpoint', checkpoint)

        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------

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

            # TRAINING
            for current_batch in range(num_train_batches_per_epoch):
                print('\nBatch {} - time: {:.2f}s'.format(current_batch, time.time() - batch_time))

                # Capture time that batch begins
                batch_time = time.time()

                # Get batch from indices
                indices = [shuffle_index[i % num_train_samples] for i in
                           range(current_batch * utils.BATCH_SIZE, (current_batch + 1) * utils.BATCH_SIZE)]
                image_id, batch_inputs, batch_text, batch_labels = train_feeder.generate_batch(index=indices)
                print('image_id: {}'.format(image_id))

                # Feed dict to model
                feed_dict = {
                    model.inputs: batch_inputs,
                    model.labels: batch_labels,
                    model.sequence_length: [model.max_char_count] * utils.BATCH_SIZE
                }
                model.is_training = True

                # Run training
                summary_str, optimiser, batch_cost, step, decoded = sess.run(
                    fetches=[model.merged_summary, model.optimiser, model.cost, model.global_step, model.decoded],
                    feed_dict=feed_dict
                )
                print('Epoch {} - Step {} - Batch_cost {} - Cost over length {}'.format(current_epoch, step, batch_cost, batch_cost / len(batch_text[0])))
                print('text_batch:', batch_text[0])
                print('decoded   :', utils.decode_result(decoded[0]))

                # Calculate cost
                delta_batch_cost = batch_cost * utils.BATCH_SIZE
                train_cost += delta_batch_cost
                train_writer.add_summary(summary=summary_str, global_step=step)

                # Save checkpoint
                if step % utils.SAVE_STEPS == 1:
                    # Make directory of not existing
                    if not os.path.isdir(utils.CRNN_CHECKPOINT_DIR):
                        os.mkdir(utils.CRNN_CHECKPOINT_DIR)
                    # Log info
                    print('Save checkpoint of step', step)
                    # Save session
                    saver.save(sess=sess,
                               save_path=os.path.join(utils.CRNN_CHECKPOINT_DIR, 'crnn-model'),
                               global_step=step)

            # ---------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------------

            # VALIDATION
            print('\n--Validation')
            # Initialise batch accuracy
            acc_batch_total = 0
            learning_rate = 0

            val_feeder = DataManager(train=False)
            print('Validation size:', val_feeder.size)

            num_val_samples = val_feeder.size
            num_val_batches_per_epoch = int(num_val_samples / utils.BATCH_SIZE)
            print('num_val_samples', num_val_samples)
            print('num_val_batches_per_epoch: {}\n'.format(num_val_batches_per_epoch))

            # Shuffle validation indices
            shuffle_index_val = np.random.permutation(num_val_samples)

            for j in range(num_val_batches_per_epoch):
                # Get batch from indices
                val_indices = [shuffle_index_val[i % num_val_samples] for i in
                               range(j * utils.BATCH_SIZE, (j + 1) * utils.BATCH_SIZE)]
                val_id, val_inputs, val_text, val_labels = val_feeder.generate_batch(index=val_indices)
                print('val_id: {} - val_text: {}'.format(val_id, val_text))

                # Validation feed dict
                val_feed_dict = {
                    model.inputs: val_inputs,
                    model.labels: val_labels,
                    model.sequence_length: [model.max_char_count] * utils.BATCH_SIZE
                }
                model.is_training = False

                # Run validation
                dense_decoded = sess.run(
                    fetches=model.decoded,
                    feed_dict=val_feed_dict
                )

                # Print decoded result
                accuracy = utils.calculate_accuracy(original_seq=val_text, decoded_seq=dense_decoded,
                                                    ignore_value=-1, is_print=False)
                print('- accuracy: {:.2f}%'.format(accuracy * 100))
                acc_batch_total += accuracy

            # Average accuracy
            accuracy = (acc_batch_total * utils.BATCH_SIZE) / num_val_samples
            avg_train_cost = train_cost / ((current_batch + 1) * utils.BATCH_SIZE)

            # Capture time epoch ends
            now = datetime.datetime.now()
            timestamp = '\n[{}/{} {}:{}:{}]'.format(now.day, now.month, now.hour, now.minute, now.second)
            epoch_info = 'Epoch {}/{}:'.format(current_epoch + 1, utils.NUM_EPOCHS)
            params_results = 'lr = {}, train_cost = {}, acc = {},'.format(learning_rate, avg_train_cost, accuracy)
            time_elapsed = 'time_elapsed = {}'.format(time.time() - start_time)
            print(timestamp, epoch_info, params_results, time_elapsed)
            print('-----')



def main(_):
    """
    Main function
    """
    if utils.NUM_GPUS == 0:
        dev = '/cpu:0'
    elif utils.NUM_GPUS == 1:
        dev = '/device:GPU:1'
    else:
        raise ValueError('Only support 0 or 1 GPU.')

    with tf.device(dev):
        if utils.MODE == 'train':
            train()


if __name__ == '__main__':
    tf.logging.set_verbosity(v=tf.logging.INFO)
    tf.app.run()
