import numpy as np
import pandas as pd
import re
import os
import utils

from scipy.misc import imread, imresize, imsave


def resize_image(image_path):
    image_arr = imread(image_path, mode='L')
    rows, cols = np.shape(image_arr)

    # If image width is bigger than defined width
    if cols > utils.IMG_WIDTH:
        cols = utils.IMG_WIDTH
        ratio = float(utils.IMG_WIDTH) / cols
        rows = int(utils.IMG_HEIGHT * ratio)
        final_arr = imresize(image_arr, size=(rows, utils.IMG_WIDTH))

    # If image width <= defined width
    else:
        final_arr = np.zeros(shape=(utils.IMG_HEIGHT, utils.IMG_WIDTH))
        ratio = float(utils.IMG_HEIGHT) / rows
        new_width = int(cols * ratio)
        image_arr_resized = imresize(image_arr, size=(utils.IMG_HEIGHT, new_width))
        final_arr[:, 0:min(utils.IMG_WIDTH, new_width)] = image_arr_resized[:, 0:utils.IMG_WIDTH]

    final_arr = np.array(final_arr).astype(np.float32) / 255.

    return final_arr


class DataManager(object):
    def __init__(self, train):
        self.train = train
        self.data = []
        self.load_data()

    @property
    def size(self):
        return len(self.data)

    def load_data(self):
        # Load dataframe
        data_df = pd.read_csv(os.path.join('dataset', 'labels.csv'))

        # Get data for train set
        if self.train:
            data_df = data_df[data_df['set'] == 'train']
        # Get data for validation set
        else:
            data_df = data_df[data_df['set'] == 'val']

        # For each row in dataframe
        for _, row in data_df.iterrows():
            image = resize_image(image_path=row['path'])
            self.data.append((
                row['id'],
                image,
                row['text'],
                utils.encode_label(label=row['text'])
            ))

    def generate_batch(self, index):
        batch_id = []
        batch_image = []
        batch_text = []
        batch_label = []

        for i in index:
            image_id, image, text, label = self.data[i]
            batch_id.append(image_id)
            batch_image.append(image)
            batch_text.append(text)
            batch_label.append(label)

        batch_image = np.swapaxes(batch_image, axis1=1, axis2=2)
        new_shape = (len(batch_image), utils.IMG_WIDTH, utils.IMG_HEIGHT, utils.IMG_CHANNEL)
        batch_image = np.reshape(np.array(batch_image), newshape=new_shape)

        batch_text = np.reshape(np.array(batch_text), newshape=(-1))

        batch_label = np.reshape(np.array(batch_label), newshape=(-1))
        batch_label = utils.sparse_tuple_from_label([batch_label])

        return batch_id, batch_image, batch_text, batch_label

