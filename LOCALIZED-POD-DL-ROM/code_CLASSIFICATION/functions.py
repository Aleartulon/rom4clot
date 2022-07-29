#functions
import numpy as np
import time
import tensorflow as tf
import datetime
import matplotlib.style as style
from mpl_toolkits import mplot3d
import scipy.io as sio
import os
import matplotlib.mlab
import matplotlib.pyplot as plt
from pylab import *
import csv
from matplotlib.colors import LogNorm

def prepare_dataset(params, labels, N, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((params,labels))
    dataset = dataset.shuffle(N)
    dataset = dataset.batch(batch_size = batch_size)
    return dataset

@tf.function
def train_step(param_batch, label_batch, Classifier, ae_optimizer):
    with tf.GradientTape() as ae_tape:
        output = Classifier(param_batch,training = True)
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_function(label_batch,output)
        trainable_vars = Classifier.trainable_variables 
    ae_grads = ae_tape.gradient(loss,trainable_vars)
    ae_optimizer.apply_gradients(zip(ae_grads,trainable_vars))
    return loss

def training(dataset_train, dataset_val, n_train, n_val, batch_size,Classifier,ae_optimizer,path, EPOCHS = 100):
    best_loss = 2

    train_loss = tf.keras.metrics.Mean('train_loss',dtype = tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss',dtype = tf.float32)
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = path+'/logs/gradient/tape/'+current_time+'/train'
    val_log_dir = path+'/logs/gradient/tape/'+current_time+'/val'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    summary_combined_first= tf.summary.create_file_writer(train_log_dir)
    summary_combined_second= tf.summary.create_file_writer(val_log_dir)

    count = 0
    for epoch in range(EPOCHS):
        print('----------------------Training-----------------------')
        iterator_train = iter(dataset_train)
        try:
            while True:
                param_batch, label_batch = iterator_train.get_next()
                loss_train = train_step(param_batch, label_batch, Classifier, ae_optimizer)
                train_loss(loss_train)
        except tf.errors.OutOfRangeError:
            pass

        with train_summary_writer.as_default():
            tf.summary.scalar('loss_train',train_loss.result(),step = epoch)
  

        print('----------------------Validation-----------------------')
        iterator_train = iter(dataset_val)
        try:
            while True:
                param_batch, label_batch = iterator_train.get_next()
                output = Classifier(param_batch)
                loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
                loss_val = loss_function(label_batch,output)
                val_loss(loss_val)
        except tf.errors.OutOfRangeError:
                pass


        if val_loss.result() < best_loss:
            best_loss = val_loss.result()
            Classifier.save_weights(path+'/weights_classifier/classifier_tf',save_format='tf')
            print('Saved')
            count = 0
        else:
            count += 1

        with val_summary_writer.as_default():
            tf.summary.scalar('loss_val', val_loss.result(),step = epoch)
        with summary_combined_first.as_default():
            tf.summary.scalar('loss_val-loss_train',train_loss.result(),step = epoch)
        with summary_combined_second.as_default():
            tf.summary.scalar('loss_val-loss_train',val_loss.result(),step = epoch)

        template = 'Epoch {}, loss_train {}, loss_val {} '
        print(template.format(epoch+1,train_loss.result(), val_loss.result()))
        train_loss.reset_states()
        val_loss.reset_states()
        print('Validation has not decreased for ' + str(count)+' epochs')
        if count == 200:
            print('Stop due to early validation')
            break







