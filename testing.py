#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

class Test_model:
    def __init__(self, test_path, model_path):
        self.test_path = test_path
        self.model_path = model_path
        
    def preprocess(self):        
        test_dataset = tf.data.Dataset.list_files(self.test_path)
        def preprocess_image(file_path):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3) # returns a tensor with the decoded pixel vals
            image = tf.image.resize(image, [150, 150])
            image /= 255.0  # normalize pixel values to [0, 1]
            return image, file_path
        # Map the preprocessing function to each image file path in the dataset
        test_dataset = test_dataset.map(preprocess_image)
    
        # Batch the dataset with a batch size of 32 (a tf.data.Dataset object)
        self.batched_dataset = test_dataset.batch(32)
        
        return self.batched_dataset
    
    def predict(self):        
        model = load_model(self.model_path)
        # Loop over the batches, save filenames make predictions
        self.predictions = []
        self.filenames = []
        # tuple unpacking
        for batch_images, batch_filenames in self.batched_dataset:
            batch_filenames = [os.path.basename(file_path.numpy().decode('utf-8')) for file_path in batch_filenames]
            yhat = model.predict(batch_images)
            self.predictions.append(yhat)
            self.filenames.extend(batch_filenames)
            
        return self.predictions, self.filenames
    
    def save(self):          
        f_names = []
        # get rid of extension
        for i in range(len(self.filenames)):
            filename = self.filenames[i]
            position = filename.index('.')
            f_name = filename[:position]
            f_names.append(f_name)
               
        # Concatenate the predictions from all batches
        predicts= np.concatenate(self.predictions, axis=0)
        predicted_classes = np.argmax(predicts, axis=1)
        print(predicted_classes.dtype)
        f_names = np.reshape(f_names, (-1, 1))
        output_data = np.column_stack((f_names, predicted_classes.astype(int)))
        header = np.array(['number', 'class'])  # Column names
        output_data = np.vstack((header, output_data))  # Add the header to the top
        np.savetxt('/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/prediction.csv', output_data, delimiter=',', fmt=('%s', '%s'))
        
        return 
        
        
        
        
        
        
        
        