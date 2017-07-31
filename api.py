from flask import Flask, render_template, request, redirect
import json, ast
import datetime
from scipy import misc
import numpy as np
from model_inference import retrieve_model

app = Flask(__name__)

"""login_manager = LoginManager()
login_manager.init_app(app)"""

app.config["DEBUG"] = True
app.config['SECRET_KEY'] = 'harekrishna'
app.config['WTF_CSRF_ENABLED'] = True

model = retrieve_model()


@app.route('/upload/', methods=['POST'])
def upload():
   if request.method == 'POST':
      if len((request.files)) == 0:
         return "No files uploaded !"
      elif len((request.files)) > 1:
	 return "Mutliple file uploads not allowed !"
      else:
      	file = request.files.values()[0]
      if file:
        data = misc.imread(file, flatten=True)
	data = np.array(data)
	if data.shape != (28, 28):
       	    data = misc.imresize(data, [28, 28])
	    data = 256. - data
	else:
	    data = data - np.min(data)
	    data = data/np.float(np.max(data))
	
	result = model.predict(data.reshape([1,28,28,1]))
	result = list(result[0])
	final_result = np.argmax(result)
	result = map("{0:.5f}".format, result)
	result = (final_result, result)
	return json.dumps(result)

if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8888)
