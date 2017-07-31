from __future__ import division, print_function, absolute_import

import tflearn
from model_training import build_network


def retrieve_model(model_path="mnist_model.tfl"):
	"""
		Load an already trained model checkpoint, restore the model using that and return the restored model object
        Parameters
        ----------
        model_path : string
        	Path to the model.
        Returns
        -------
        model : <class 'tflearn.models.dnn.DNN'>
            The restored model object
    """
	# Building convolutional network skeleton
	network = build_network()
	model = tflearn.DNN(network, tensorboard_verbose=0)

	# Loading the model checkpoint
	model.load(model_path, weights_only=True)

	return model
