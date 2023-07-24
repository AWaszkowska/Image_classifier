# Image_classifier
Image classification using CNN and MLP (from scratch), that allows user to try different inputs and even datasets by creating object instances in main file. Keras, Tensorflow, Numpy libraries are used. There are two class files (MLP and CNN), and object instances are created in main.py file, to allow the user to input parameters like name, num_labels, num_neurons, num_layers, optimizer, training, testing and validation set. None of these are hard-coded. 

Results like accuracy graph through epochs, loss function history graph, precision and recall are saved automatically to csv file to enable seamless analysis. 

File load_preprocess for data reading and batching using keras and preprocessing like normalization of pixel values to 0-1.
MLP just for experiment. It is usually not suitable for image classification tasks, because of flat input.
CNN incorporates selected number of convolutional layers, corresponding numper of max pooling layers, one flatten layer and selected number of dense layers, with restriction of max 3 layers (if exceeded, throws an error).

Keras Sequential API is used to create both networks.
