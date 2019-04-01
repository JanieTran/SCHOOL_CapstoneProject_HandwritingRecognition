import os
import tarfile
import urllib
import sys
import time
import glob
import pickle
import xml.etree.ElementTree as ET
import cv2
import json
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import logging


# --------------------------------------------------------------------
# ----- HELPER FUNCTIONS
# --------------------------------------------------------------------

def crop_image(image, bounding_box):
    """
    Crop image by bounding box in percentages
    :param image: input image of shape [h, w]
    :param bounding_box: bounding box to crop
    :return: cropped image
    """
    # Get box coordinates and size from input
    (x, y, w, h) = bounding_box

    # Get box coordinates and size w.r.t image
    x = x * image.shape[1]
    y = y * image.shape[0]
    w = w * image.shape[1]
    h = h * image.shape[0]

    # Cropped image
    (x1, y1, x2, y2) = (x, y, x + w, y + h)
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))

    return image[y1:y2, x1:x2]


def resize_image(image, desired_size):
    """
    Resize image while keeping aspect ratio
    :param image: image to resize
    :param desired_size: output size
    :return:
        - image: resized image
        - crop_bounding_box: (x, y, w, h) w.r.t resized image
    """
    # Get height and width of original image
    size = image.shape[:2]

    # If shrinking image
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        # Get aspect ratio for each dimension
        ratio_w = float(desired_size[0] / size[0])
        ratio_h = float(desired_size[1] / size[1])
        # Take the smaller one
        ratio = min(ratio_w, ratio_h)
        # Calculate new size based on the ratio
        new_size = tuple([int(dim * ratio) for dim in size])
        # Resize image using OpenCV
        image = cv2.resize(src=image, dsize=(new_size[1], new_size[0]))
        size = image.shape

    # Get difference in dimension
    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])

    # Length in pixels of border at each side
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Color of added border
    color = image[0][0]
    if color < 230:
        color = 230

    # Add border to image
    image = cv2.copyMakeBorder(src=image, top=top, bottom=bottom,
                               left=left, right=right,
                               borderType=cv2.BORDER_CONSTANT,
                               value=float(color))

    # New bounding box
    box_x = left / image.shape[1]
    box_y = top / image.shape[0]
    box_w = (image.shape[1] - right - left) / image.shape[1]
    box_h = (image.shape[0] - bottom - top) / image.shape[0]
    crop_bounding_box = (box_x, box_y, box_w, box_h)

    # TODO: what for?
    image[image > 230] = 255

    return image, crop_bounding_box


def crop_handwriting_page(image, bounding_box, image_size):
    """
    Crop image based on bounding box
    :param image: input image
    :param bounding_box: bounding box in percentages to crop
    :param image_size: size to scale output image
    :return: cropped image of size image_size
    """
    # Crop image
    image = crop_image(image=image, bounding_box=bounding_box)
    # Resize image
    image, _ = resize_image(image=image, desired_size=image_size)

    return image


# --------------------------------------------------------------------
# ----- HELPER FUNCTIONS
# --------------------------------------------------------------------


class IAMDataset():
    """
    Class to obtain IAM dataset
    :param train: whether to load the training or testing set
    :param output_type: data type of output, either text or bounding box
    :param ouput_text_as_list: whether ouput text will be a list of lines string
    """
    # Helper constants
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)

    def __init__(self, dataset_option, train=True, output_text_as_list=False):
        options = ['form_text', 'form_box', 'line']
        error = 'UNEXPECTED DATASET OPTION: {} not in {}'.format(dataset_option, options)
        assert dataset_option in options, error

        # URL for data
        url = 'http://www.fki.inf.unibe.ch/DBs/iamDB/data/{data_type}/{filename}.tgz'
        if dataset_option == 'line':
            self._data_urls = [url.format(data_type='lines', filename='lines')]
        else:
            self._data_urls = [url.format(data_type='forms', filename='forms' + a)
                               for a in ['A-D', 'E-H', 'I-Z']]

        # URL for XML
        self._xml_url = 'http://www.fki.inf.unibe.ch/DBs/iamDB/data/xml/xml.tgz'

        # Credentials
        with open(file='credentials.json') as f:
            credentials = json.load(f)
        self._credentials = (credentials['username'], credentials['password'])

        # Train flag
        self._train = train

        # Image data file name
        image_data_file_name = 'image-{}*.plk'.format(dataset_option)
        self.image_data_file_name = os.path.join('iamdataset', image_data_file_name)

        # Create data directory if not exist
        self._root = 'iamdataset'
        if not os.path.isdir(self._root):
            os.makedirs(self._root)
        self._output_text_as_list = output_text_as_list

        data = self._get_data()

    @staticmethod
    def _report_hook(count, block_size, total_size):
        """
        Prints a process bar compatible with urllib.request.urlretrieve
        """
        toolbar_width = 40
        percentage = float(count * block_size) / total_size * 100

        # Taken from https://gist.github.com/sibosutd/c1d9ef01d38630750a1d1fe05c367eb8
        sys.stdout.write('\r')
        sys.stdout.write('Completed: [{:{}}] {:>3)%'.format(
            '-' * int(percentage / (100. / toolbar_width)),
            toolbar_width, int(percentage)
        ))
        sys.stdout.flush()

    def _extract(self, archive_file, archive_type, output_dir):
        """
        Extract archived files
        :param archive_file: path to file
        :param archive_type: type of file, tar or zip
        :param output_dir: path to extract files
        """
        print('Extracting', archive_file)

        # Check if legit archive type
        _types = ['tar', 'zip']
        error = 'UNEXPECTED ARCHIVE TYPE: {} not in {}'.format(archive_type, _types)
        assert archive_type in _types, error

        # Path to extract destination
        dest = os.path.join(self._root, output_dir)

        # If tar file
        if archive_type == 'tar':
            tar = tarfile.open(name=archive_file, mode='r:gz')
            tar.extractall(path=dest)
            tar.close()
        # If zip file
        else:
            zip_ref = zipfile.ZipFile(file=archive_file, mode='r')
            zip_ref.extractall(path=dest)
            zip_ref.close()

    def _download(self, url):
        """
        Download data using credentials
        :param url: url of the file to download
        """
        # Password manager
        password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, url, self._credentials[0], self._credentials[1])

        # Authentication handler
        handler = urllib.request.HTTPBasicAuthHandler(password_manager)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

        # Open URL
        opener.open(url)
        filename = os.path.basename(url)
        print('Downloading', filename)

        urllib.request.urlretrieve(url, reporthook=self._report_hook,
                                   filename=os.path.join(self._root, filename))[0]
        sys.stdout.write('\n')
