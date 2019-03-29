import numpy as np
import os
import utils

from PIL import Image


def get_input_lens(sequences):
    """
    Get length of all sequences
    :param sequences: input sequences
    :return:
    """
    # 64 is the output channels of the last layer of CNN
    lengths = np.asarray([utils.MAX_STEPSIZE for _ in sequences], dtype=np.int64)
    return sequences, lengths


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
    shape = np.asarray([len(sequences), np.asarray(indices).max(axis=0)[1] + 1], dtype=np.int64)

    return indices, values, shape


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
                im = np.array(Image.open(fp=img_name).convert('L')).astype(np.float32) / 255.
                # Reshape image to predefined size
                im = np.reshape(im, [utils.IMG_HEIGHT, utils.IMG_WIDTH, utils.IMG_CHANNEL])

                # Append image to class properties
                self.image.append(im)

                # Append label to class properties
                label = img_name.split('/')[-1].split('_')[1].split('.')[0]

                label = [utils.SPACE_INDEX if label == utils.SPACE_TOKEN else utils.ENCODE_MAPS[c] for c in list(label)]
                self.labels.append(label)


    @property
    def size(self):
        """
        Number of images iterable
        :return: number of images
        """
        return len(self.labels)

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
