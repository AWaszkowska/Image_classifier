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
name = 'adam_l2'
kernel_regularizer=regularizers.L2(0.0005)
optimizer='adam'

train, val, test = load_preprocess.load_preprocess(path_train, num_labels)

cn = cnn.CNN()
cn.build_model(name, num_labels, train, val, test, 4, 64, 2,kernel_regularizer, optimizer)

mp = mlp.MLP()
mp.build_model('mp', 6, 512, 3, 'adam', train, val, test)

test_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/seg_test/*.jpg'
model_path = '/home/amelia97/Documents/Python/aml/AML-2023-Project2_alunos/prject_two/mp.h5'
test_mlp = testing.Test_model(test_path, model_path)
t_prepro = test_mlp.preprocess()
t_predict = test_mlp.predict()
test_mlp.save()