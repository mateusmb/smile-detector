"""
LeNet Convolutional Neural Networks by [LeCun et al., 1998]

This CNN was designed for handwritten and machine-printed character
recognition. (http://yann.lecun.com/exdb/lenet/)

The architecture of this CNN is as following:

INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
(As seen in Rosebrock, 2017, adapted)
"""

"""
Imports
note: 'backend as K' is for Keras backend configuration
"""
from keras                      import backend      as K
from keras.models               import Sequential
from keras.layers.core          import Dense
from keras.layers.core          import Flatten
from keras.layers.core          import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class LeNet:
    """
    The LeNet class builds a LeNet model object
    """

    def __init__(self, width, height, depth, classes):
        """
        Constructor of LeNet class

        :param width: width of input image
        :param height: height of input image
        :param depth: number of channel of input images
        :param classes: number of class labels for classification
        """
        self.model = Sequential()

        # Initializing input shape ordering acording with backend"""
        if K.image_data_format() == "channels_first":
            self.input_shape = (depth, height, width)
        else:
            self.input_shape = (height, width, depth)

        self.classes = classes


    def build(self):
        """
        Builds the model with LeNet architecture
        """

        # 1 - CONV => RELU => POOL
        self._buildConvReluPoolLayers(filters=20, filter_size=5,
                                      padding_type="same",
                                      activation_function="relu",
                                      pooling_size=2, stride_size=2,
                                      first_layer=True)


        # 2 - CONV => RELU => POOL
        self._buildConvReluPoolLayers(filters=50, filter_size=5,
                                      padding_type="same",
                                      activation_function="relu",
                                      pooling_size=2, stride_size=2)


        # 3 - FC => RELU
        self._buildFCLayers(nodes=500, activation_function="relu",
                            flatten=True)

        # 4 - FC with softmax classifier
        self._buildFCLayers(nodes=self.classes, activation_function="softmax")


    def compile(self, loss_function, optimizer, metrics_list=None):
        """
        Function that compiles the model with regularization methods

        :param loss_function: Function for loss computation
        :param optimizer: Optimizer to backpropagate
        :param metrics_list: The list of metrics to be added
        """
        self.model.compile(loss=loss_function, optimizer=optimizer,
                           metrics=metrics_list)


    def trainNetwork(self, data, labels, validation_data=None,
                     batch_size=None, class_weight=None, epochs=1, verbose=1):
        """
        Function that trains (fit) the model built

        :param data: the data to be fitted
        :param labels: the output labes
        :param validation_data: tuple for validate the fitting
        :param batch_size: the size of fitting batches
        :param class_weight: weight of class to manage imbalance
        :param epochs: the number of fitting epochs
        :param verbose: verbosity of the network
        :return: Return a History object of model fitting
        """
        return self.model.fit(data, labels, validation_data=validation_data,
                              class_weight=class_weight,
                              batch_size=batch_size, epochs=epochs,
                              verbose=verbose)


    def predict(self, data, batch_size=None):
        """
        Predicts labels based on data using the trained network

        :param data: the data to predict the labels
        :param batch_size: number of samples by gradient update
        :return: predictions of type numpy array
        """
        if batch_size is not None:
            return self.model.predict(data, batch_size=batch_size)
        else:
            return self.model.predict(data)

    def save(self, path):
        """
        Serializes the model in path

        :param path: the path where the model will be saved
        """
        self.model.save(path)

    def _buildConvReluPoolLayers(self, filters, filter_size,
                                 padding_type, activation_function,
                                 pooling_size, stride_size, first_layer=False):
        """
        Internal method for building models with CONV, ReLu activation and
        POOL layers.

        :param filters: number of filters to learn
        :param filter_size: the size of the filters
        :param padding_type: the type of padding
        :param activation_function: function to use for activation
        :param pooling_size: size of POOL square matrix
        :param stride_size: size of stride square matrix
        :param first_layer: flag to add input shape, default is False
        """
        if first_layer:
            self.model.add(Conv2D(filters,
                              (filter_size, filter_size),
                              padding=padding_type,
                              input_shape=self.input_shape))
        else:
            self.model.add(Conv2D(filters,
                                  (filter_size, filter_size),
                                  padding=padding_type))

        self.model.add(Activation(activation_function))
        self.model.add(MaxPooling2D(pool_size=(pooling_size, pooling_size),
                                    strides=(stride_size, stride_size)))


    def _buildFCLayers(self, nodes, activation_function, flatten=False):
        """
        Internal method for building models with Fully Connected Layers.

        :param nodes: number of nodes to Dense layers
        :param activation_function: function to use for activation
        :param flatten: flag for 1D array, default is False
        """
        if flatten:
            self.model.add(Flatten())

        self.model.add(Dense(nodes))
        self.model.add(Activation(activation_function))