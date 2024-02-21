# from keras.layers import Input, Dense, Activation
# from keras.layers import Maximum, Concatenate

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf


class mqttGAN():
    '''
    def __init__(self, filename='D:/workspace/GAN/swGAN/data/process_mqtt_data.npz'):
        pass
    
    def load_data(self):
        data = np.load(self.filename)
        xmal, ymal, xben, yben = data['x_mal_processed'], data['y_mal'], data['x_normal_processed'], data['y_normal']
        print('sw--success def data_load!')
        return (xmal, ymal), (xben, yben)
    '''
    
    # data load
    filename='D:/workspace/GAN/swGAN/data/process_mqtt_data.npz'
    data = np.load(filename)
    
    # split data
    xmal, ymal, xben, yben = data['x_mal_processed'], data['y_mal'], data['x_normal_processed'], data['y_normal']
    
    xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.20)
    xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.20)
   
    # train data
    
    
if __name__ == '__main__':
    mqttgan = mqttGAN