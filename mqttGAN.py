'''
restart on 2024-03-18
origin mnistGAN to mqttGAN
'''

################################
# 공통 패키지 불러오기
################################
import numpy as np
import pandas as pd
from PIL import Image
import math
import os

from keras import models, layers, optimizers
from keras.datasets import mnist
import keras.backend as K
from sklearn.model_selection import train_test_split

print(K.image_data_format)

import tensorflow as tf


def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1, 2, 3))


def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1, 2, 3))


################################
# GAN 모델링
################################
class GAN(models.Sequential):
    def __init__(self, input_dim=32):  # input_dim = args.n_train = 32
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

        self.compile_all()

    def compile_all(self):
        # Compiling stage
        d_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(loss=mse_4d_tf, optimizer="SGD")
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(7 * 7 * 128, activation='tanh'))  # H, W, C = 7, 7, 128
        model.add(layers.BatchNormalization())
        # The Conv2D op currently only supports the NHWC tensor format on the CPU.
        model.add(layers.Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def DISCRIMINATOR(self):
        # The Conv2D op currently only supports the NHWC tensor format on the CPU.
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',
                                input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))

    def train_both(self, x):
        ln = x.shape[0]
        # First trial for training discriminator
        z = self.get_z(ln)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x, w))
        y2 = np.array([1] * ln + [0] * ln).reshape(-1, 1)  # Necessary!
        d_loss = self.discriminator.train_on_batch(xw, y2)

        # Second trial for training generator
        z = self.get_z(ln)
        self.discriminator.trainable = False
        g_loss = self.train_on_batch(z, np.array([1] * ln).reshape(-1, 1))
        self.discriminator.trainable = True

        return d_loss, g_loss


################################
# GAN 학습하기
################################
'''
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]  # (1,2) for NHWC
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[:, :, 0]  # NHWC
    return image
'''

def get_x(X_train, index, BATCH_SIZE):
    return X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

'''
def save_images(generated_images, output_fold, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_fold + '/' +
        str(epoch) + "_" + str(index) + ".png")


def load_data(n_train):
    (X_train, y_train), (_, _) = mnist.load_data()
    return X_train[:n_train]
'''
def load_data2(filename):
    data = np.load(filename)
    xmal, ymal, xben, yben = data['xmal'], data['ymal'], data['xben'], data['yben']

    return (xmal, ymal), (xben, yben)

def hex_to_int(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        if value.startswith('0x'):
            try:
                return int(value, 16)
            except ValueError:
                return value
        else:
            try:
                return int(value, 16)
            except ValueError:
                return value
    else:
        return value

def convert_hex_to_int(df):
    df['mqtt.protoname'] = df['mqtt.protoname'].map({'MQTT':1, '0':0})

    for column in df.columns:
        df[column] = df[column].apply(hex_to_int)
    return df


def padding_data(data, data_type):
    # data = pd.read_csv(f)

    # print(data.head())

    # print("##### processing #####")
    # data = data.drop('target', axis=1)
    # print(data.head())

    # 데이터 프레임 열 개수 확인
    num = data.shape[1]  # 3
    # print('origin_data column : ', num)

    # 80개 열을 가진 데이터 프레임 생성
    new_data = pd.DataFrame()

    # new_line 생성
    for i in range(num):
        new_data[data.columns[i]] = data[data.columns[i]]

    for i in range(num, 80):
        new_data['new_col_' + str(i - num)] = 0  # 새로운 열에 0 채우기

    new_num = data.shape[1]  # 80
    # print(new_data)
    process_data = 'D:/workspace/GAN/swGAN/data/data_' + data_type + '_make_80_colums.csv'
    # print(file_name)

    # print('##### padding data size : ', new_data.shape, '#####')
    # 저장
    new_data.to_csv(process_data, index=False)
    return new_data


def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim
    n_train = args.n_train

    print('args : ', args)
    # Namespace(batch_size=16, epochs=1000, input_dim=10, n_train=32, output_fold='GAN_OUT')

    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    # X_train = load_data(n_train)
    # print('type : ', type(X_train)) # <class 'numpy.ndarray'>
    # print('shape : ', X_train.shape) # (32, 28, 28)

    num_of_normal = 1000
    num_of_mal = 1000

    f_ben = 'D:\\workspace\\GAN\\swGAN\\data\\xben(24.03.18).csv'
    f_mal = 'D:\\workspace\\GAN\\swGAN\\data\\xmal(24.03.18).csv'

    xben = pd.read_csv(f_ben)
    yben = np.zeros(num_of_normal)
    xmal = pd.read_csv(f_mal)
    ymal = np.zeros(num_of_mal)

    # hex 값 int로 변환
    xben = convert_hex_to_int(xben)
    xmal = convert_hex_to_int(xmal)

    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    ## The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW
    ## X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:]) # <-- NCHW format
    # X_train = X_train.reshape(X_train.shape + (1,))  # <-- NHWC format
    # 채널 차원 추가
    # 딥러닝 프레임워크에서 요구되는 데이터 형식에 일치시키기 위하여

    xben = padding_data(xben, 'ben')
    yben = np.zeros(num_of_normal)
    xmal = padding_data(xmal, 'mal')
    ymal = np.ones(num_of_mal)

    # print('after reshape : ', X_train.shape) # (32, 28, 28, 1)
    xben = xben.to_numpy()
    xmal = xmal.to_numpy()

    xben = xben.astype(np.float64)
    xmal = xmal.astype(np.float64)

    print("xben.shape=", xben.shape)  # (1000,80)
    print("yben.shape=", yben.shape)  # (1000,)
    print("xmal.shape=", xmal.shape)  # (1000,80)
    print("ymal.shape=", ymal.shape)  # (1000,)

    # train_size_a = 0.2
    # train_size_a = 0.4
    # train_size_a = 0.6
    train_size_a = 0.8

    xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)
    xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, train_size=train_size_a,
                                                                    test_size=0.20, shuffle=False)

    print("xtrain_mal.shape=", xtrain_mal.shape)  # (800, 19)
    print("xtest_mal.shape=", xtest_mal.shape)  # (200,80)
    print("ytrain_mal.shape=", ytrain_mal.shape)  # (800,)
    print("ytest_mal.shape=", ytest_mal.shape)  # (200,)

    gan = GAN(input_dim) # input_dim = 10

    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        if epoch % 10 == 0:
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

    # gan.generator.save_weights(output_fold + '/' + 'generator', True)
    # gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)

    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)


################################
# GAN 예제 실행하기
################################
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for the networks')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Epochs for the networks')
    parser.add_argument('--output_fold', type=str, default='GAN_OUT',
                        help='Output fold to save the results')
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Input dimension for the generator.')
    parser.add_argument('--n_train', type=int, default=32,
                        help='The number of training data.')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()