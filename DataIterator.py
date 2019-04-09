import numpy as np
import pandas as pd
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


class DataIterator:
    def __init__(self, train):
        self.image = []
        self.labels = []
        self.text = []
        self.image_id = []

        # Walkthrough data dir to get all folders and files
        # for root, sub_folder, file_list in os.walk(data_dir):
        #     # For each image file found
        #     for file_path in file_list:
        #         # Get the file name
        #         img_name = os.path.join(root, file_path)
        #
        #         # Convert image to B&W and then to np array, normalise to between 0 and 1
        #         im = np.array(Image.open(fp=img_name).convert('L')).astype(np.float32) / 255.
        #         # Reshape image to predefined size
        #         im = np.reshape(im, [utils.IMG_HEIGHT, utils.IMG_WIDTH, utils.IMG_CHANNEL])
        #
        #         # Append image to class properties
        #         self.image.append(im)
        #
        #         # Append label to class properties
        #         label = img_name.split('/')[-1].split('_')[1].split('.')[0]
        #
        #         label = [utils.SPACE_INDEX if label == utils.SPACE_TOKEN else utils.ENCODE_MAPS[c] for c in list(label)]
        #         self.labels.append(label)

        # Load dataframe
        data_df = pd.read_csv(os.path.join('dataset', 'labels.csv'))

        # Get data for train set
        if train:
            data_df = data_df[data_df['set'] == 'train']
        # Get data for validation set
        else:
            data_df = data_df[data_df['set'] == 'val']

        # For each row in dataframe
        count = 0
        for _, row in data_df.iterrows():
            # Convert image to B&W
            im = Image.open(fp=row['path']).convert('L')
            # Resize to (60, 800)
            im = im.resize((utils.IMG_WIDTH, utils.IMG_HEIGHT))
            # Normalise between 0 and 1
            im = np.array(im).astype(np.float32) / 255.
            # Reshape image to predefined size
            im = np.reshape(im, [utils.IMG_HEIGHT, utils.IMG_WIDTH, utils.IMG_CHANNEL])

            # Append image to class properties
            self.image.append(im)
            self.image_id.append(row['id'])

            # Append label to class properties
            text = row['text']
            label = [utils.SPACE_INDEX if text == utils.SPACE_TOKEN else utils.ENCODE_MAPS[c] for c in list(text)]
            self.text.append(text)
            self.labels.append(label)

            count += 1
            if count == 5 and not train:
                break

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
            image_id = [self.image_id[i] for i in index]
            text_batch = [self.text[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels
            image_id = self.image_id
            text_batch = self.text

        batch_inputs, batch_seq_lens = get_input_lens(np.array(image_batch))
        batch_labels = utils.sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_lens, batch_labels, image_id, text_batch
