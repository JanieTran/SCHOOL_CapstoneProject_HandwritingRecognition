import numpy as np
import os
from PIL import Image
from utils import FLAGS, sparse_tuple_from_label


def get_input_lens(sequences):
    """
    Get length of all sequences
    :param sequences: input sequences
    :return:
    """
    # 64 is the output channels of the last layer of CNN
    lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype=np.int64)
    return sequences, lengths


class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []

        # Walkthrough data dir to get all folders and files
        for root, sub_folder, file_list in os.walk(data_dir):
            # For each image file found
            for file_path in file_list:
                # Get the file name
                img_name = os.path.join(root, file_path)

                # Convert image to B&W and then to np array, normalise to between 0 and 1
                im = np.array(Image.open(fp=img_name).convert('L')).astype(np.float32) / 255
                # Reshape image to predefined size
                im = np.reshape(im, [FLAGS.img_height, FLAGS.img_width, FLAGS.img_channel])

                # Append image to class properties
                self.image.append(im)

    @property
    def size(self):
        """
        Number of images iterable
        :return: number of images
        """
        return len(self.image)

    def get_labels(self, indices):
        """
        Get labels
        :param indices: indices of desired labels
        :return: labels
        """
        labels = []
        for i in indices:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        """
        Generate image batches
        :param index: indices of images to be chosen, or None for all images
        :return: sequences in batch, length of each sequence, labels in batch
        """
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        batch_inputs, batch_seq_lens = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_lens, batch_labels
