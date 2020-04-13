import os
import cv2
import imutils
import numpy as np
from   imutils                   import paths
from   keras.preprocessing.image import img_to_array


class DatasetLoader:
    """
    Class DatasetLoader reads data and labels from given path
    """
    def __init__(self, data_path):
        """
        Constructor

        :param data_path: the path of dataset
        """
        self.data = []
        self.labels = []
        self.data_path = data_path

    def load(self, verbose=-1, image_width=28, label_name_position=-3):
        """
        Function to load the dataset from given path

        :param verbose: verbosity level, will print information each verbose
        :param image_width:  option width to resize images, default is 28
        :param label_name_position: position where is the label name in path
        :return: a tuple of numpy arrays data and labels
        """
        image_paths = sorted(list(paths.list_images(self.data_path)))
        for (i, image_path) in enumerate(image_paths):
            image = self._loadImage(image_path, image_width)
            self.data.append(image)
            label = self._extractLabel(image_path, label_name_position)
            self.labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

        self.data   = np.array(self.data, dtype="float") / 255.0
        self.labels = np.array(self.labels)
        return (self.data, self.labels)

    def _loadImage(self, image_path, image_width):
        """
        Internal method to load image using OpenCV

        :param image_path: the path of the image which will be loaded
        :param image_width: the width to resize image
        :return: image loaded with OpenCV
        """
        image = cv2.imread(image_path)
        image = self._preprocess(image, image_width)
        return image

    def _preprocess(self, image, image_width):
        """
        Internal method to preprocess the image

        :param image: the image to be preprocessed
        :param image_width: the width to resize image
        :return: the image preprocessed
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=image_width)
        image = img_to_array(image)
        return image

    def _extractLabel(self, image_path, label_name_pos):
        """
        Internal method to extract the labels from path

        :param image_path: the path of the image
        :param label_name_pos: position where to extract label from path
        :return: a label String
        """
        label = image_path.split(os.path.sep)[label_name_pos]
        label = "smiling" if label == "positives" else "not_smiling"
        return label
