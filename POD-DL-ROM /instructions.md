# What to do in order
1) Go to code/get_data.ipynb and follow the procedures as it is explained in the comments. At the end you will have a directory POD-DL-ROM/MODELS/specie/interval/N/Data as explained in the code.
2) Send to the cluster the directory POD-DL-ROM/MODELS/specie/interval/N/Data. In the cluster you should have the exact copy of the structure of POD-DL-ROM. Be careful since some directories may not be constructed in the cluster.
3) In the cluster, go to POD-DL-ROM/code and run main_training.py. Be careful to choose well all the quantities to be specified in main_training.py. At the and of the training you will have a directory 'POD-DL-ROM/MODELS/specie/interval/N/my_directory', whose name 'my_directory' you specify in main_training.py. There you will find the Encoder, DFNN and Decoder weights.
4) Import on your laptop the directory POD-DL-ROM/MODELS/specie/interval/N/my_directory, paying attention to put in in the right place.
5) Go to POD-DL-ROM/code and run the Test.ipynb code to test.

NOTA BENE:
In order to work you need in the directory POD-DL-ROM 3 files: The DoE file with the simulations, the cell centers file with the mesh coordinates and the X_LHS_Uniform.csv file with the parameters of every simulation.