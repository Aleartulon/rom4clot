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



def loss_function(inp_enc,DFNN_out, encode_out, decoder_out, N_h,omega_h = 0.5, omega_n = 0.5):

    inp_enc = tf.reshape(inp_enc,[-1,int(N_h)])
    decoder_out = tf.reshape(decoder_out,[-1,int(N_h)])
    inp_enc = tf.cast(inp_enc,float32)
    loss_n = omega_h*tf.math.reduce_mean(tf.math.reduce_sum(tf.math.pow(encode_out-DFNN_out,2),axis = 1))
    loss_h = omega_n*tf.math.reduce_mean(tf.math.reduce_sum(tf.math.pow(inp_enc-decoder_out,2),axis = 1))
    return (loss_n,loss_h)


@tf.function
def train_step(inp_enc,inp_dfnn,Encoder,DFNN,Decoder,omega_h ,omega_n, ae_optimizer,N_h):
    with tf.GradientTape() as ae_tape:
        encode_out = Encoder(inp_enc,training = True)
        DFNN_out = DFNN(inp_dfnn, training = True)
        decoder_out = Decoder(DFNN_out, training = True)
        
        loss = loss_function(inp_enc, DFNN_out, encode_out, decoder_out,N_h, omega_h, omega_n)
        loss_sum = loss[0]+loss[1],
        trainable_vars = Encoder.trainable_variables + DFNN.trainable_variables+ Decoder.trainable_variables 
    ae_grads = ae_tape.gradient(loss_sum,trainable_vars)
    ae_optimizer.apply_gradients(zip(ae_grads,trainable_vars))
    return loss

def prepare_dataset(S, params, n_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((S,params))
    dataset = dataset.shuffle(n_train)
    dataset = dataset.batch(batch_size=batch_size)
    return dataset

def training(dataset_train, dataset_val, n_train, n_val, batch_size,Encoder, DFNN,Decoder,omega_h,omega_n,ae_optimizer,path, N_h, EPOCHS = 100):
    best_loss = 2

    train_loss_n = tf.keras.metrics.Mean('train_loss_n', dtype = tf.float32)
    train_loss_h = tf.keras.metrics.Mean('train_loss_h', dtype = tf.float32)
    train_loss_tot = tf.keras.metrics.Mean('train_loss_tot', dtype = tf.float32)
    val_loss_tot = tf.keras.metrics.Mean('val_loss_tot', dtype = tf.float32)
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
                batch_inp_enc ,batch_inp_dfnn = iterator_train.get_next()
                batch_inp_enc = tf.reshape(batch_inp_enc,[-1,int(np.sqrt(N_h)),int(np.sqrt(N_h)),1])
                loss = train_step(batch_inp_enc, batch_inp_dfnn,Encoder, DFNN,Decoder,omega_h,omega_n,ae_optimizer,N_h)
                train_loss_n(loss[0])
                train_loss_h(loss[1])
                train_loss_tot(loss[0]+loss[1])
        except tf.errors.OutOfRangeError:
            pass

        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss_train_n',train_loss_n.result(),step = epoch)
            tf.summary.scalar('loss_train_h',train_loss_h.result(),step = epoch)
            tf.summary.scalar('loss_train_tot',train_loss_tot.result(),step = epoch)
            
        print('----------------------Validation-----------------------')
        iterator_val = iter(dataset_val)
        try:
            while True:
                batch_inp_enc ,batch_inp_dfnn = iterator_val.get_next()
                batch_inp_enc = tf.reshape(batch_inp_enc,[-1,int(np.sqrt(N_h)),int(np.sqrt(N_h)),1])

                encode_out = Encoder(batch_inp_enc)
                DFNN_out = DFNN(batch_inp_dfnn)
                decoder_out = Decoder(DFNN_out)

                loss = loss_function(batch_inp_enc, DFNN_out, encode_out, decoder_out, N_h,omega_h, omega_n)
                val_loss_tot(loss[0]+loss[1])
        except tf.errors.OutOfRangeError:
            pass
        if val_loss_tot.result() < best_loss:
            best_loss = val_loss_tot.result()
            Encoder.save_weights(path+'/weights_Encoder/Encoder_tf',save_format='tf')
            DFNN.save_weights(path+'/weights_DFNN/DFNN_tf',save_format='tf')
            Decoder.save_weights(path+'/weights_Decoder/Decoder_tf',save_format='tf')
            print('Saved')
            count = 0
        else:
            count+=1

            

        with val_summary_writer.as_default():
            tf.summary.scalar('loss_val_tot',val_loss_tot.result(),step = epoch)

        with summary_combined_first.as_default():
            tf.summary.scalar('loss_val_tot-loss_train_tot',train_loss_tot.result(),step = epoch)
        with summary_combined_second.as_default():
            tf.summary.scalar('loss_val_tot-loss_train_tot',val_loss_tot.result(),step = epoch)

        template = 'Epoch {}, loss_train_n {}, loss_train_h {}, loss_train_tot {}, loss_val_tot {} '
        print(template.format(epoch+1,train_loss_n.result(), train_loss_h.result(), train_loss_tot.result(), val_loss_tot.result()))
        train_loss_n.reset_states()
        train_loss_h.reset_states()
        train_loss_tot.reset_states()
        val_loss_tot.reset_states()
        print('Validation loss has not decreased for '+str(count)+' steps')
        if count == 500:
            print('Stop due to early validation')
            break



