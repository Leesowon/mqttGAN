import pandas as pd
import numpy as np
import os

import keras.backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split

'''
ben_origin_data = pd.read_csv("D:/workspace/GAN/swGAN/data/mqttdataset_legitimate.csv")
mal_origin_data = pd.read_csv("D:/workspace/GAN/swGAN/data/mqttdataset_malicious.csv")

# 상위 10000개 데이터 선택 후 .csv로 저장
xben = ben_origin_data.head(10000)
xben.to_csv('D:/workspace/GAN/swGAN/data/mqttdataset_legitimate_10000.csv', index=False)
xmal = mal_origin_data.head(10000)
xmal.to_csv('D:/workspace/GAN/swGAN/data/mqttdataset_malicious_10000.csv', index=False)

# data feature 별 형식 확인
# print(ben_data.dtypes)

# data 형식 확인
print('xben shape = ', xben.shape)
print('xmal shape = ', xmal.shape)
'''

'''
    # target 열에 대해 레이블 매핑 (ben : 0, mal : 1)
    ben_data['label'] = ben_data['target'].map({'legitimate': 0})
    mal_data['label'] = mal_data['target'].map({'dos': 1, 'slowite': 1, 'bruteforce': 1, 'malformed': 1, 'flood': 1})
    

    # 매핑 후 데이터를 CSV 파일로 저장
    ben_data.to_csv('D:\workspace\GAN\swGAN\data\ben_data_labeling.csv', index=False)
    mal_data.to_csv('D:\workspace\GAN\swGAN\data\mal_data_labeling.csv', index=False)
'''
# GAN 모델링
from keras import models, layers, optimizers

def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1,2,3))

def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1,2,3))


class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        """
        self, self.generator, self.discriminator are all models
        """
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.blackbox_detector = self.build_blackbox_detector()

        self.compile_all()
        
        
        
        
        
    
def train(args):
    
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim
    n_train = args.n_train

    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    # X_train = load_data(n_train)
    # print("X_train.shape=",X_train.shape)
    
    num_of_normal = 10000
    num_of_mal = 10000
    xben = pd.read_csv('D:/workspace/GAN/swGAN/data/mqttdataset_legitimate_10000.csv')
    yben = np.zeros(num_of_normal) # 레이블 : 정상 0 
    xmal = pd.read_csv('D:/workspace/GAN/swGAN/data/mqttdataset_malicious_10000.csv')
    ymal = np.ones(num_of_mal) # 레이블 : 악성 1
    
    print("xben.shape=", xben.shape)
    print("yben.shape=", yben.shape)
    print("xmal.shape=", xmal.shape)
    print("ymal.shape=", ymal.shape)
    # print("xmal[0]=", xmal[0])
    
    print('========== reshape(2->3) ==========')
    '''
    'xben'과 'xmal'이 CSV 파일이라면, 
    먼저 이들을 pandas DataFrame으로 불러온 후, numpy 배열로 변환
    '''
    # xben_reshaped = xben.reshape(-1, 2, 17)
    # xmal_reshaped = xmal.reshape(-1, 2, 17)
   
    # DataFrame을 numpy 배열로 변환 후 reshape
    xben_reshaped = xben.values.reshape(-1, 2, 17)
    xmal_reshaped = xmal.values.reshape(-1, 2, 17)
    print("xben_reshaped.shape=", xben_reshaped.shape)
    print("xmal_reshaped.shape=", xmal_reshaped.shape)
    
    
    '''
    print('========== data split ==========')
    train_size = 0.8
    xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, train_size = train_size, test_size=0.20, shuffle=False)
    xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, train_size = train_size, test_size=0.20, shuffle=False)

    print("xtrain_mal.shape=", xtrain_mal.shape)
    print("xtest_mal.shape=", xtrain_mal.shape)
    print("ytrain_mal.shape=", xtrain_mal.shape)
    print("ytest_mal.shape=", xtrain_mal.shape)
    
    print("xtrain_ben.shape=", xtrain_mal.shape)
    print("xtest_ben.shape=", xtrain_mal.shape)
    print("ytrain_ben.shape=", xtrain_mal.shape)
    print("ytest_ben.shape=", xtrain_mal.shape)
    '''
    
    
    
    

    
    # 정규화 -> train : test = 8 : 2
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5 # 정규화(-1~1)
    # X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    '''
    >> 이미지 데이터의 형상을 변환
    신경망이 처리하는 데이터의 차원 수에 따라 입력 데이터의 형상을 맞춰주어야 하기 때문
    예를 들어, 합성곱 신경망(CNN)은 일반적으로 4차원 데이터를 입력으로 받으므로, 3차원 이미지 데이터에 채널 차원을 추가하여 
    4차원으로 만들기 위해 이런 형상 변환이 필요
    '''
'''
    gan = GAN(input_dim)

    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            x = get_x(X_train, index, BATCH_SIZE)

            d_loss, g_loss = gan.train_both(x)

            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            z = gan.get_z(x.shape[0])
            w = gan.generator.predict(z, verbose=0)
            save_images(w, output_fold, epoch, 0)

        d_loss_ll.append(d_loss_l)
        g_loss_ll.append(g_loss_l)

    gan.generator.save_weights(output_fold + '/' + 'generator', True)
    gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)

    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)
    '''
    
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the networks')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for the networks')
    parser.add_argument('--output_fold', type=str, default='GAN_OUT',
        help='Output fold to save the results')
    parser.add_argument('--input_dim', type=int, default=80, help='Input dimension for the generator.')

    parser.add_argument('--n_train', type=int, default=32,
        help='The number of training data.')
    
    args = parser.parse_args()
    train(args)
    
if __name__ == '__main__':
    main()