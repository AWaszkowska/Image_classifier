#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

def load_preprocess(path, num_labels):
     """
     

     Parameters
     ----------
     path : path of the folder containing all training images,
            in subfloders named as labels
     num_labels : number of labels in the training dataset

     Returns
     -------
     train, validation and test dataset of shuffled images.
     Also plots a sample of 4 loaded images.

     """
     
     
     data = tf.keras.utils.image_dataset_from_directory(path, image_size=(150,150))
     data_iterator = data.as_numpy_iterator()
     batch = data_iterator.next()

     fig, ax = plt.subplots(ncols=4, figsize=(20,20))
     for idx, img in enumerate(batch[0][:4]):
         ax[idx].imshow(img.astype(int))
         ax[idx].title.set_text(batch[1][idx])

     # as we see, the image is in standard color numbers 0-255
     # for better generalization we should scale to 0-1

     data = data.map(lambda x, y: (x/255, tf.one_hot(y, depth=num_labels)))
     data.as_numpy_iterator().next()

     # split data to train and validation set
     train_size =  int(len(data)*.7)
     val_size = int(len(data)*.2)
     test_size = int(len(data)*.2)
     # shuffle the data
     data_shuffled = data.shuffle(buffer_size=len(data), seed=1)
     train = data_shuffled.take(train_size)
     val = data_shuffled.skip(train_size).take(val_size)
     test = data_shuffled.skip(train_size+val_size).take(test_size)
     
     return train, val, test