# digit-recognition-service
A digit recognition service that predicts the label of an uploaded digit image

CNN (Convolutional Neural Network) is used as the ML model architecture and the implementation is derived from the tflearn mnist example code. Tensorflow (at the lower level) and tflearn (at the higher level) are used as the maachine learning platforms omn the implementation side. 

The model is first trained until 20 epochs and then saved / checkpointed, and then can be used to make inferences for new unseen digit image(s).

Images should contain a digit and can be variably dimensioned and channeled (RGB, grayscale, etc.). The uploaded input image to the api is first converted to a 28X28 grayscale image with normalized pixel values, which is then fed to the CNN classifier as input. The CNN classifier returns probabilities for the 10 classes (digits 0 to 9 respectively). The ap returns a tuple with the first entry being the preducted label (i.e. argmax of the outputted list) and the second entry being the ordered list of probability itself

Flask is used as the web framework to receive POST requests in the form of image uploads (ONE IMAGE PER REQUEST) and returns the result (the tuple described in the above paragraph) in the form of a json string.

Image pre-processing needs further work. The current version is limited to two family of images (i) 28 X 28 grayscale, (ii) Bigger RGB images with lighter backround and darker digit pixels. Machine learning model could be trained further, or an even better model could be researched/chosen, but CNN is a fairly standard literature choice for MNIST modelling.

The model checkpoint files currently visible in this repository should ideall y be removed from here, and placed on a separate database or aws s3, but is kept here for the sake of reproducibility of results if needed.

The service is hosted on an aws ec2 machine, and is accessible at the follwoing endpoint: http://52.209.79.236:8888/upload/
One needs to make a POST request and upload one image file at a time / per request. And should expect a response that looks like: [8, ["0.00001", "0.00000", "0.00000", "0.00000", "0.00000", "0.00000", "0.00021", "0.00000", "0.99938", "0.00040"]]
Where, 8 is the digit/label that is predicted for the image, and the rest is an ordered list (0 to 9) of probabilities of the image belonging to the respective digit-class. If you upload multiple image files, you should expect this response: "Mutliple file uploads not allowed !" , and is you don't upload any files, then : "No files uploaded !".
