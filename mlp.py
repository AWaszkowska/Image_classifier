#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from tensorflow import keras

class MLP:
    def __init__(self):
        pass
    
    def build_model(self, name, num_labels, num_neurons, num_layers, optimizer, train, val, test):
        

        self.name = name
        self.num_labels = num_labels
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.train = train
        self.val = val
        self.test = test

        
        model = Sequential()
        model.add(Flatten(input_shape=(150,150,3)))
        model.add(Dense(self.num_neurons, activation='relu', input_shape=(150,150,3)))
        for i in range(self.num_layers-1):
            if self.num_neurons>128:
                model.add(Dense(self.num_neurons, activation='relu'))
                self.num_neurons = self.num_neurons//2
        model.add(Dense(self.num_labels, activation='softmax'))
          
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        model.summary()

        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = model.fit(self.train, epochs=1, validation_data=self.val, callbacks=[tensorboard_callback])
        hist.history 
        
        # plot performance

        fig = plt.figure()
        plt.plot(hist.history['loss'], color='teal', label='loss')
        plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc='upper left')
        plt.show

        fig = plt.figure()
        plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
        plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc = "upper left")
        plt.show()
        
        # evaluate performance
        
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()

        len(self.test)
        for batch in self.test.as_numpy_iterator():
            X, y = batch
            yhat = model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
            
        self.precision = pre.result().numpy()
        self.recall = re.result().numpy()
        self.acc = acc.result().numpy()

        print(pre.result().numpy())
        # write performance to csv
        stats_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/proj2/stats/test_performance'
        output_data = [self.name, self.precision, self.recall, self.acc]

        with open(stats_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Precision', 'Recall', 'Accuracy'])
            writer.writerow(output_data)

        # save the model
        
        model_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/' + self.name + '.h5'
        model.save(model_path)
        
        return model