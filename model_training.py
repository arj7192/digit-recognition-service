from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist


def build_network():
    """
        Prepare the network architecture skeleton and return the network graph tensor object
        Parameters
        ----------
        None
        Returns
        -------
        network : <class 'tensorflow.python.framework.ops.Tensor'>
        	The symbolic representation of the network in form of tensorflow tensor object.
    """
    # Building convolutional network
    network = input_data(shape=[None, 28, 28, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')
    return network


def load_data():
    """
        Load the standard mnist data
        Parameters
        ----------
        None
        Returns
        -------
        tuple of X, Y, testX and testY : <tuple>
            Tuple of X, Y (training data pairs), testX and testY (testing data pairs) in that order.
    """
    # Data loading and preprocessing
    X, Y, testX, testY = mnist.load_data(one_hot=True)
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    return X, Y, testX, testY


if __name__ == "__main__":
    # Training
    network = build_network()
    X, Y, testX, testY = load_data()
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': Y}, n_epoch=1,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='convnet_mnist')

    # Saving the model
    model.save("mnist_model.tfl")
