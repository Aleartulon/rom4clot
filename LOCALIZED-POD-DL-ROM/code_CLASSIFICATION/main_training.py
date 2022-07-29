import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from architecture import Classifier
from functions import*

EPOCHS = 1000
n_clusters = 3
n_neurons = 50
n_layers_dfnn = 4
N = 5 #POD basis size for clustering
batch_size = 20
specie = 'vWFs'
n_data = 3800
n_train = int(n_data*0.8)
n_val = n_data-n_train
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
ae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
dir = '3_cluster'
home = '../'+specie+'/CLASSIFICATION/MODELS/'+str(N)+'/Data/'
path = '../'+specie+'/CLASSIFICATION/MODELS/'+str(N)+'/'+dir

params = np.loadtxt(open(home+'param.csv', "rb"), delimiter=',')
idxs = np.random.permutation(params.shape[0])
params = params[idxs]
params_train, params_val = params[:n_train], params[n_train:]
template = 'M Loaded! Shape = {}'
print(template.format(np.shape(params)))
del params

labels = np.loadtxt(open(home+'labels.csv', "rb"), delimiter=',')
labels = labels[idxs]
labels_train, labels_val = labels[:n_train], labels[n_train:]
template = 'Labels Loaded! Shape = {}'
print(template.format(np.shape(labels)))
del labels

Classifier = Classifier(n_layers = n_layers_dfnn,neurons = n_neurons,length_output = n_clusters)

dataset_train = prepare_dataset(params_train, labels_train, n_train, batch_size)
dataset_val = prepare_dataset(params_val, labels_val, n_val, batch_size)

training(dataset_train, dataset_val, n_train, n_val, batch_size, Classifier,ae_optimizer,path,EPOCHS)
Classifier.summary()
