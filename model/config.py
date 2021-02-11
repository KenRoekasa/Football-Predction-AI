# Config file for the model of the neural network
import time
import numpy as np
import tensorflow as tf



INPUT_DIM=50

network_topology=[90]

learning_rate = 0.001
batch_size = 2
time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
NAME = 'model-%s-learningrate-%f-batch-size-%d' % (time,learning_rate,batch_size)
opt = tf.keras.optimizers.SGD(lr=learning_rate)





def combination_of_means(teama_mean_array,teamb_mean_array): # How to combine the two means find the difference or put them all into the input
    mean_data_array = np.append(teama_mean_array, teamb_mean_array)
    # mean_data_array = teama_mean_array - teamb_mean_array
    return mean_data_array


