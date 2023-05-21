#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.metrics import Precision, Recall, BinaryAccuracy
import csv

class CNN:
    def __init__(self):
        pass
    
    def build_model(self, name, num_labels, train, val, test, num_conv, num_filt, num_dense, kernel_regularizer=None, optimizer='adam'):
        """
        

        Parameters
        ----------
        name : name of the model, how you want to save it
        num_conv : number of Conv2D layers
        num_filt : number of filters in Conv2D layers, should be a power of 2
        num_dense : number of Dense layers, except the last one: choose 1, 2 or 3
        kernel_regularizer : a parameter of Conv2D layer, l2 is advised such as l2(0.0005)
        optimizer: optimizer used on model

        Returns
        -------
        model, plots of loss, val_loss, accuracy, val_accuracy and metrics on testing set
        """
        
        self.name = name
        self.num_conv = num_conv
        self.num_filt = num_filt
        self.num_dense = num_dense
        self.kernel_regularizer = kernel_regularizer
        self.optimizer = optimizer
        
        
        model = Sequential()
        
        if self.kernel_regularizer is not None:
            model.add(Conv2D(self.num_filt, (3,3), activation='relu', input_shape=(150,150,3), kernel_regularizer=self.kernel_regularizer))
            model.add(MaxPooling2D(2,2))
            for i in range(self.num_conv-1):
                if self.num_filt>16:
                    model.add(Conv2D(self.num_filt, (3,3), activation='relu', kernel_regularizer=self.kernel_regularizer))
                    model.add(MaxPooling2D(2,2))
                    self.num_filt = self.num_filt/2
            model.add(Flatten())
        else:
            model.add(Conv2D(self.num_filt, (3,3), activation='relu', input_shape=(150,150,3)))
            model.add(MaxPooling2D(2,2))
            for i in range(self.num_conv-1):
                if self.num_filt>16:
                    model.add(Conv2D(self.num_filt, (3,3), activation='relu'))
                    model.add(MaxPooling2D(2,2))
                    self.num_filt = self.num_filt/2
            model.add(Flatten())
        if (self.num_dense == 1):
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
        elif (self.num_dense == 2):
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(256, activation='relu'))
        elif (self.num_dense == 3):
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(256, activation='relu'))
        else:
            raise ValueError("Sir, we running outta computing powers!")
        if num_labels == 2:
            model.add(Dense(self.num_labels, activation='sigmoid'))
            loss_func = tf.losses.BinaryCrossentropy()
        else:
            model.add(Dense(num_labels, activation='softmax'))   
            loss_func = tf.losses.CategoricalCrossentropy()
            
        model.compile(optimizer=self.optimizer, loss=loss_func, metrics=['accuracy'])
        model.summary()

        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        hist = model.fit(train, epochs=15, validation_data=val, callbacks=[tensorboard_callback])
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

        len(test)
        for batch in test.as_numpy_iterator():
            X, y = batch
            yhat = model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
            
        print(pre.result(), re.result(), acc.result())
        
        self.precision = pre.result()
        self.recall = re.result()
        self.acc = acc.result()
        
        # write performance to csv
        # stats_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/stats/' + self.name
        # output_data = np.hstack(self.name, self.precision, self.recall, self.acc)
        # np.savetxt(stats_path, output_data, delimiter=',', fmt=('%s'))
        
        stats_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/proj2/stats/test_performance'
        output_data = [self.name, self.precision, self.recall, self.acc]

        with open(stats_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Precision', 'Recall', 'Accuracy'])
            writer.writerow(output_data)

        # save the model
        
        model_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/' + self.name + '.h5'
        model.save(model_path)
        
        return model, self.precision, self.recall, self.acc 

        