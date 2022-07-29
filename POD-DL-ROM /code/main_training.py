#this script should be run with GPU. It goes together with architecture.py and functions.py. It needs as data to run the files 
#Input.csv and M_train.csv, which must be located in '../MODELS/'+specie+'/'+str(N), which are created by the notebook 
#get_data. If the files you want to test are in another directory remember to modify 'home' and 'path'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from architecture import Encoder
from architecture import DFNN
from architecture import Decoder
from functions import*


EPOCHS = 10000 #maximum number of epochs
N = 64 #number of modes
latent_space = 6 #usually this is equal to the number of parameters +1
INPUT_SHAPE_ENC = (np.sqrt(N),np.sqrt(N),1)
INPUT_SHAPE_DFNN = (latent_space,1)
n_neurons = 100 #neurons for each layer of DFNN
n_layers_dfnn = 4 #layers of DFNN
omega_h = 0.5 #weight in loss function
omega_n = 0.5
batch_size = 25
n_data = 3800 #size data used for the training
n_train = int(n_data*0.8)
n_val = n_data-n_train #divide training and validation
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
ae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
dir = '10000_kernel_7_batch25' #name of directory with outputs
specie = 'vWFs'
interval = 'full' #if you are not training indipendently more clusters just put 'full'
home = '../MODELS/'+specie+'/'+interval+'/'+str(N) #this is where the two input files are saved
path = home+'/'+dir #this is where you want the outputs of the training to be saved (the Encoder, DFNN and Decoder models)


##load normalized matrices S and M 
S = np.loadtxt(open(home+'/Data/'+"Input.csv", "rb"), delimiter=',')
template = 'S Loaded! Shape = {}'
print(template.format(np.shape(S)))
idxs = np.random.permutation(S.shape[0])
S = S[idxs]
S_train, S_val = S[:n_train, :], S[n_train:, :]
del S
params = np.loadtxt(open(home+'/Data/'+'M_train.csv', "rb"), delimiter=',')
params = params[idxs]
params_train, params_val = params[:n_train], params[n_train:]
template = 'M Loaded! Shape = {}'
print(template.format(np.shape(params)))
del params


Encoder = Encoder(input_shape=(np.sqrt(N),np.sqrt(N),1),num_of_layers = 4,length_output = latent_space)
DFNN = DFNN(n_layers = n_layers_dfnn,input_shape=(latent_space,1),neurons = n_neurons,length_output = latent_space)
Decoder = Decoder(input_kernel=latent_space,num_of_layers = 4, N = N)

dataset_train = prepare_dataset(S_train, params_train, n_train, batch_size)
dataset_val = prepare_dataset(S_val, params_val, n_val, batch_size)

training(dataset_train, dataset_val, n_train, n_val, batch_size,Encoder, DFNN,Decoder,omega_h,omega_n,ae_optimizer,path,N,EPOCHS)

Encoder.summary()
DFNN.summary()
Decoder.summary()




