from flask import Flask, request
import json
from scipy import misc
import numpy as np
from model_inference import retrieve_model

app = Flask(__name__)
app.config["DEBUG"] = True
app.config['WTF_CSRF_ENABLED'] = True

# Load the pre-trained model
model = retrieve_model()


# Define the upload endpoint functionality
@app.route('/upload/', methods=['POST'])
def upload():
	"""
		Pre-process an uploaded image file and feed it to the machine learning model to get the probabilities, and
		return those output probabilities tupled with the most likely class label.
		Parameters
		----------
		None
		Returns
		-------
		result : string
			json-stringified tuple (a, b) where a is the most likely predicted digit (between 0 and 9) and b is the
			ordered list of probabilities for each digit.
	"""
	if request.method == 'POST':
		if len((request.files)) == 0:
			return "No files uploaded !"
		elif len((request.files)) > 1:
			return "Mutliple file uploads not allowed !"
		else:
			file = request.files.values()[0]
		if file:
			data = misc.imread(file, flatten=True)
			result = model.predict(pre_process_data(data))
			result = list(result[0])
			final_result = np.argmax(result)
			result = map("{0:.5f}".format, result)
			result = json.dumps(final_result, result)
			return result
	else:
		return "Please make a POST request !"


def pre_process_data(data):
	"""
        Pre-process the image data matrix into a 28X28 normalized input feature matrix for the machine learning model
        Parameters
        ----------
        data: list
        	Basically the output of the imread function applied on the image file.
        Returns
        -------
        data : numpy array
        	A pre-processed, normalized, and reshaped (into size [1, 28, 28, 1]) matrix
    """
	data = np.array(data)
	if data.shape != (28, 28):
		data = misc.imresize(data, [28, 28])
		data = 256. - data
	data = data - np.min(data)
	data = data / np.float(np.max(data))
	return data.reshape([1, 28, 28, 1])


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8888)
