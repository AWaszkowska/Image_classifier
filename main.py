#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cnn
import load_preprocess
import mlp
import testing
from tensorflow import keras
from keras import regularizers

path_train = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/seg_train'
num_labels = 6
name = 'sgd_l2'
kernel_regularizer=regularizers.L2(0.0005)
# kernel_regularizer=None
optimizer='sgd'

train, val, test = load_preprocess.load_preprocess(path_train, num_labels)

cn = cnn.CNN()
cn.build_model(name, num_labels, train, val, test, 6, 128, 2,kernel_regularizer, optimizer)

mp = mlp.MLP()
mp.build_model('mp', 6, 512, 3, 'adam', train, val, test)

test_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/seg_test/*.jpg'
model_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/adam_l2.h5'
test = testing.Test_model(test_path, model_path)
test.preprocess()
test.predict()
test.save()