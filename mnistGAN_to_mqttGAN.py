from keras.datasets import mnist
import numpy as np
import pandas as pd
# from PIL import Image
import math
import os

import keras.backend as K
import tensorflow as tf

import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.layers import Embedding, LSTM, Dense, Dropout, Activation, Convolution1D, MaxPooling1D

K.set_image_data_format('channels_first')
print(K.image_data_format)

# GAN 모델링
from keras import models, layers, optimizers
# 수정
from tensorflow.keras.layers import Conv2D


def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1,2,3))

def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1,2,3))


class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.blackbox_detector = self.build_blackbox_detector()

        self.compile_all()


    def build_blackbox_detector(self):
        model = models.Sequential()
        model.add(Embedding(35000, 512, input_length=100))
        model.add(Dropout(0.25))

        model.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='valid',
                                activation='relu', subsample_length=1))
        model.add(MaxPooling1D(pool_length=4))

        model.add(LSTM(256))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model


    def compile_all(self):
        # Compiling stage
        d_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(loss=mse_4d_tf, optimizer="SGD")
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
        self.blackbox_detector.compile(loss='binary_crossentropy', optimizer=d_optim)


    def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()

        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 5 * 1, activation='tanh'))

        model.add(layers.BatchNormalization())

        model.add(layers.Reshape((128, 5, 1), input_shape=(128 * 5 * 1,)))

        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='tanh'))

        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (3, 3), padding='same', activation='tanh'))

        return model


    def DISCRIMINATOR(self):
        model = models.Sequential()

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='tanh', input_shape=(1, 10, 8)))

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model


    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))


    def train_both(self, xmal, ymal, xben, yben, epoch, BATCH_SIZE, period):
        debug = 0
        old_TRR = 1.0

        xben_len = xben.shape[0]
        xmal_len = xmal.shape[0]

        z = self.get_z(BATCH_SIZE)
        if (debug == 1): print("z.shape=", z.shape)

        xmal_reshape = np.reshape(xmal, (xmal.shape[0], 80))

        xmal_reshape_noise = z  # only z

        w = self.generator.predict(xmal_reshape_noise, verbose=0)

        w_reshape = np.reshape(w, (w.shape[0], 80))

        for i in range(xmal_reshape.shape[0]):
            for j in range(int(80 / period)):
                w_reshape[i][j*period+(period-1)] = xmal_reshape[i][j]

                w = np.reshape(w_reshape, (w.shape[0], 1, 10, 8))

                if (debug == 1): print("w.shape=", w.shape)

                xw = np.concatenate((xben, w))

                y2 = [0] * xben_len + [1] * xmal_len

                d_loss_real = self.discriminator.train_on_batch(w, ymal)
                d_loss_fake = self.discriminator.train_on_batch(xben, yben)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                self.discriminator.trainable = False

                z = self.get_z(BATCH_SIZE)

                xmal_reshape = np.reshape(xmal, (xmal.shape[0], 80))

                xmal_reshape_noise = z  # only z

                w = self.generator.predict(xmal_reshape_noise, verbose=0)

                w_reshape = np.reshape(w, (w.shape[0], 80))

                for i in range(xmal_reshape.shape[0]):
                    for j in range(int(80 / period)):
                        w_reshape[i][j * period + (period - 1)] = xmal_reshape[i][j]
                w = np.reshape(w_reshape, (w.shape[0], 1, 10, 8))

                g_loss = self.train_on_batch(w_reshape, [0] * xmal_len)

                self.discriminator.trainable = True

                return d_loss, g_loss


################################
# GAN 학습하기
################################
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def get_x(X_train, index, BATCH_SIZE):
    return X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]


def save_images(generated_images, output_fold, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_fold + '/' +
        str(epoch) + "_" + str(index) + ".png")


def load_data(n_train):
    (X_train, y_train), (_, _) = mnist.load_data()

    return X_train[:n_train]


def load_data2(filename):
    data = np.load(filename)
    xmal, ymal, xben, yben = data['xmal'], data['ymal'], data['xben'], data['yben']

    return (xmal, ymal), (xben, yben)


def process_data_set2(fpath):
    sequence = []
    with open(fpath) as f:
        for line in f:
            try:
                # split = line.split(",")
                line = line.strip()  # 문자열 끝의 불필요한 문자제거
                split = line.split(" ")
                # print(len(split))

            except Exception:
                print("Exception")

            sequence.append(split)

    seq_arr = np.array(sequence)
    # print(seq_arr.shape)

    return seq_arr

def padding_data(f, data_type):
    data = pd.read_csv(f)
    # print(data.head())

    print("##### processing #####")

    data = data.drop('target', axis=1)
    # print(data.head())

    # 데이터 프레임 열 개수 확인
    num = data.shape[1]  # 33
    # print('origin_data column : ', num)

    # 80개 열을 가진 데이터 프레임 생성
    new_data = pd.DataFrame()

    # new_line 생성
    for i in range(num):
        new_data[data.columns[i]] = data[data.columns[i]]

    for i in range(num, 80):
        new_data['new_col_' + str(i - num)] = 0  # 새로운 열에 0 채우기

    # print(new_data)

    new_num = data.shape[1]  # 80
    print('data column : ', new_data)
    process_data = 'D:/workspace/GAN/swGAN/data/data_' + data_type + '_make_80_colums.csv'
    # print(file_name)
    
    # print('##### padding data size : ', new_data.shape, '#####')
    # 저장
    new_data.to_csv(process_data, index=False)


def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim
    n_train = args.n_train

    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    # 시퀀스 한줄의 길이는 80
    num_of_normal = 1000
    num_of_mal = 1000

    f_ben = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_legitimate_1000.csv"
    f_mal = "D:\\workspace\\GAN\\swGAN\\data\\mqttdataset_malicious_1000.csv"

    xben = padding_data(f_ben, 'ben')
    yben = np.zeros(num_of_normal)
    xmal = padding_data(f_mal, 'mal')
    ymal = np.ones(num_of_mal)

    # xben = padding_data("D:\\workspace\\GAN\\swGAN\\data\\ben_data_labeling.csv")
    # yben = np.zeros(num_of_normal)
    # xmal = padding_data("D:\\workspace\\GAN\\swGAN\\data\\mal_data_labeling.csv")
    # ymal = np.ones(num_of_mal)

    print("xben.shape=", xben.shape)
    print("xmal.shape=", xmal.shape)
    print("xmal[0]=", xmal[0])

    # train_size_a = 0.2
    # train_size_a = 0.4
    # train_size_a = 0.6
    train_size_a = 0.8

    # xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.20, shuffle=False)
    # xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.20, shuffle=False)
    xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, train_size=train_size_a, test_size=0.20,
                                                                    shuffle=False)
    xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, train_size=train_size_a, test_size=0.20,
                                                                    shuffle=False)

    print("xtrain_mal.shape=", xtrain_mal.shape)
    # print("X_train.shape[0]=", X_train.shape[0])
    # print(X_train.shape[1:])

    # print("xtest_mal=", xtest_mal[0])

    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    """xtrain_mal = (xtrain_mal.astype(np.float32) - 125000.0) / 125000.0
    xtrain_mal = xtrain_mal.reshape((xtrain_mal.shape[0], 1) + xtrain_mal.shape[1:])
    xtrain_ben = (xtrain_ben.astype(np.float32) - 125000.0) / 125000.0
    xtrain_ben = xtrain_ben.reshape((xtrain_ben.shape[0], 1) + xtrain_mal.shape[1:])

    xtest_mal = (xtest_mal.astype(np.float32) - 125000.0) / 125000.0
    xtest_mal = xtest_mal.reshape((xtest_mal.shape[0], 1) + xtest_mal.shape[1:])
    xtest_ben = (xtest_ben.astype(np.float32) - 125000.0) / 125000.0
    xtest_ben = xtest_ben.reshape((xtest_ben.shape[0], 1) + xtest_ben.shape[1:])"""

    xtrain_mal = (xtrain_mal.astype(np.float32) - 35000.0) / 35000.0
    xtrain_mal = xtrain_mal.reshape((xtrain_mal.shape[0], 1) + xtrain_mal.shape[1:])
    xtrain_ben = (xtrain_ben.astype(np.float32) - 35000.0) / 35000.0
    xtrain_ben = xtrain_ben.reshape((xtrain_ben.shape[0], 1) + xtrain_mal.shape[1:])

    xtest_mal = (xtest_mal.astype(np.float32) - 35000.0) / 35000.0
    xtest_mal = xtest_mal.reshape((xtest_mal.shape[0], 1) + xtest_mal.shape[1:])
    xtest_ben = (xtest_ben.astype(np.float32) - 35000.0) / 35000.0
    xtest_ben = xtest_ben.reshape((xtest_ben.shape[0], 1) + xtest_ben.shape[1:])

    # xtest_mal_int = (xtest_mal * 125000.0 + 125000.0).astype(np.int)
    # print("xtest_mal_int[0]=", xtest_mal_int[0])

    # X_train = xmal
    # X_train = X_train.reshape(X_train.shape[0], 1, 8, 16)
    # xtrain_mal2 = xtrain_mal.reshape(xtrain_mal.shape[0], 1, 8, 16)
    # xtest_mal2 = xtest_mal.reshape(xtest_mal.shape[0], 1, 8, 16)
    # xtrain_ben2 = xtrain_ben.reshape(xtrain_ben.shape[0], 1, 8, 16)
    # xtest_ben2 = xtest_ben.reshape(xtest_ben.shape[0], 1, 8, 16)

    # xtrain_mal2 = xtrain_mal.reshape(xtrain_mal.shape[0], 1, 40, 20)
    # xtest_mal2 = xtest_mal.reshape(xtest_mal.shape[0], 1, 40, 20)
    # xtrain_ben2 = xtrain_ben.reshape(xtrain_ben.shape[0], 1, 40, 20)
    # xtest_ben2 = xtest_ben.reshape(xtest_ben.shape[0], 1, 40, 20)

    xtrain_mal2 = xtrain_mal.reshape(xtrain_mal.shape[0], 1, 10, 8)
    xtest_mal2 = xtest_mal.reshape(xtest_mal.shape[0], 1, 10, 8)
    xtrain_ben2 = xtrain_ben.reshape(xtrain_ben.shape[0], 1, 10, 8)
    xtest_ben2 = xtest_ben.reshape(xtest_ben.shape[0], 1, 10, 8)

    # xtrain_mal3 = xtrain_mal2.reshape(xtrain_mal2.shape[0], 800)
    # xtrain_ben3 = xtrain_ben2.reshape(xtrain_ben2.shape[0], 800)
    # xtest_mal3 = xtest_mal2.reshape(xtest_mal2.shape[0], 800)
    # xtest_ben3 = xtest_ben2.reshape(xtest_ben2.shape[0], 800)

    # print("X_train.shape=", X_train.shape)
    print("xtrain_mal[0]=", xtrain_mal[0])

    ################################################################################
    print("\nstart GAN()")
    gan = GAN(input_dim)

    d_loss_ll = []
    g_loss_ll = []
    st = time.time()

    min_TRR = 1.0
    min_epoch = -1
    min_test_TRR = 1.0
    min_test_epoch = -1
    Train_TRR, Test_TRR = [], []
    # period = 2
    # period = 3
    for epoch in range(epochs):

        if ((epoch + 1) % 10 == 0):  print("\n\nEpoch is", epoch + 1)
        # print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []

        ###################################################################################
        idx_mal = np.random.randint(0, xtrain_mal.shape[0], BATCH_SIZE)
        xmal_batch = xtrain_mal2[idx_mal]
        ymal_batch = ytrain_mal[idx_mal]

        idx_ben = np.random.randint(0, xtrain_ben.shape[0], BATCH_SIZE)
        xben_batch = xtrain_ben2[idx_ben]
        yben_batch = ytrain_ben[idx_ben]

        # period = 1  # 1개마다 원본삽입 (즉 모두 원본)
        # period = 2     # 2개마다 원본데이터 삽입 (2개마다 페이크데이터)
        # period = 3     # 3개마다 원본데이터 삽입 (1개 원본, 2개 페이크)
        # period = 4    # 4개마다 원본데이터 삽입 (1개 원본, 3개 페이크)
        period = 5  # 5개마다 원본데이터 삽입 (1개원본, 4개 페이크)
        # period = 10

        # d_loss, g_loss = gan.train_both(xmal_batch, ymal_batch, xben_batch, yben_batch, epoch)
        # d_loss, g_loss = gan.train_both(xmal_batch, ymal_batch,
        #                                xben_batch, yben_batch, epoch, BATCH_SIZE)
        d_loss, g_loss = gan.train_both(xmal_batch, ymal_batch,
                                        xben_batch, yben_batch, epoch, BATCH_SIZE, period)
        # print("d_loss=", d_loss)
        # print("g_loss=", g_loss)

        # et = time.time()
        # print("Elapsed time=", (et-st))

        ##############################################################################
        # Compute Train TPR
        z = gan.get_z(xtrain_mal.shape[0])
        # xtrain_mal_reshape = np.reshape(xtrain_mal, (xtrain_mal.shape[0], 128))
        # xtrain_mal_reshape = np.reshape(xtrain_mal, (xtrain_mal.shape[0], 800))
        xtrain_mal_reshape = np.reshape(xtrain_mal, (xtrain_mal.shape[0], 80))

        # sprint("xtrain_mal_reshape[0]=", xtrain_mal_reshape[0])
        # xtrain_mal_reshape_noise = xtrain_mal_reshape + z
        xtrain_mal_reshape_noise = z  # only z
        # print("xtrain_mal_reshape_noise.shape=", xtrain_mal_reshape_noise.shape)

        w = gan.generator.predict(xtrain_mal_reshape_noise, verbose=0)
        # print("w.shape=", w.shape)
        # wr = np.reshape(w, (w.shape[0],800))
        wr = np.reshape(w, (w.shape[0], 80))

        # post-noise :: xmal : (x1,x2,...,xn) , w : (w1,w2,...,wn), w' : (x1,w1,x2,w2,...,x(n/2),w(n/2))
        """for i in range(xtrain_mal_reshape.shape[0]):
            #for j in range(400):
                #w_reshape[i][j * 2] = xmal_reshape[i][j]
            #    wr[i][j*2] = xtrain_mal_reshape[i][j]
            for j in range(int(800/3)):
                #w_reshape[i][j*3] = xmal_reshape[i][j]
                wr[i][j*3] = xtrain_mal_reshape[i][j]
        #w = np.reshape(w_reshape, (w.shape[0], 1, 40, 20))
        """

        for i in range(xtrain_mal_reshape.shape[0]):
            # for j in range(400):
            #    w_reshape[i][j*2] = xmal_reshape[i][j]
            # for j in range(int(800 / period)):
            for j in range(int(80 / period)):
                wr[i][j * period + (period - 1)] = xtrain_mal_reshape[i][j]

        # w_reshape = np.reshape(w, (w.shape[0], 1, 40, 20))
        # w_reshape = np.reshape(wr, (w.shape[0], 1, 40, 20))
        w_reshape = np.reshape(wr, (w.shape[0], 1, 10, 8))

        # print("w_reshape[0]=", w_reshape[0])
        # w_reshape = w_reshape.astype(np.int32)
        # print("w_reshape.shape=", w_reshape.shape)
        TRR = gan.discriminator.evaluate(w_reshape, ytrain_mal)
        Train_TRR.append(TRR)
        # print("\nTRR=", TRR)
        # predictions = gan.discriminator.predict(w_reshape, ytrain_mal)
        # print(predictions)

        if (TRR < min_TRR):
            min_TRR = TRR
            min_epoch = epoch + 1

            # store fake train data
            # xtest_int = (xtest_mal_reshape_noise * 125000.0 + 125000.0).astype(np.int)
            # xtest_int = (wr * 125000.0 + 125000.0).astype(np.int)
            # xtrain_int = (wr * 125000.0 + 125000.0).astype(np.int)
            xtrain_int = (wr * 35000.0 + 35000.0).astype(np.int)

            # print(xtest_int[0])
            # f = open("fake_unigram_1000_train_data.txt", "w")
            # f = open("fake_train_mqtt.csv", "w")
            f = open("fake_train_mqtt_" + str(period) + ".csv", "w")

            # for j in range(xtest_int.shape[0]):
            for j in range(xtrain_int.shape[0]):
                line = ""
                # for i in range(800):
                for i in range(80):
                    # print(noise_int[i])
                    # if (i != 799):
                    if (i != 79):
                        line += str(xtrain_int[j][i]) + ","
                    else:
                        line += str(xtrain_int[j][i]) + "\n"
                f.write(line)
            f.close()

        if ((epoch + 1) % 10 == 0):
            print("\nTrain TRR=", TRR)
            print("min_Train_TRR=", min_TRR)
            print("min_epoch=", min_epoch)
            # print(f1)

        ####################################################################################
        # Compute Test TPR
        z = gan.get_z(xtest_mal.shape[0])
        # xtest_mal_reshape = np.reshape(xtest_mal, (xtest_mal.shape[0], 128))
        # xtest_mal_reshape = np.reshape(xtest_mal, (xtest_mal.shape[0], 800))
        xtest_mal_reshape = np.reshape(xtest_mal, (xtest_mal.shape[0], 80))
        xtest_mal_reshape_noise = xtest_mal_reshape + z

        # xmal : (x1, x2, ... , xn), xmal + noise : (x1, a1, x2, a2, ... , x[n/2], a[n/2])
        # for i in range(xtest_mal_reshape.shape[0]):
        #    for j in range(400):
        #        z[i][j * 2] = xtest_mal_reshape[i][j]
        xtest_mal_reshape_noise = z  # only z

        w = gan.generator.predict(xtest_mal_reshape_noise, verbose=0)
        # wr = np.reshape(w, (w.shape[0], 800))
        wr = np.reshape(w, (w.shape[0], 80))

        # post-noise :: xmal : (x1,x2,...,xn) , w : (w1,w2,...,wn), w' : (x1,w1,x2,w2,...,x(n/2),w(n/2))
        """for i in range(xtest_mal_reshape.shape[0]):
            #for j in range(400):
                # w_reshape[i][j * 2] = xmal_reshape[i][j]
                #wr[i][j * 2] = xtest_mal_reshape[i][j]
            for j in range(int(800/3)):
                #w_reshape[i][j*3] = xmal_reshape[i][j]
                wr[i][j*3] = xtest_mal_reshape[i][j]
        # w = np.reshape(w_reshape, (w.shape[0], 1, 40, 20))
        """

        for i in range(xtest_mal_reshape.shape[0]):
            # for j in range(400):
            #    w_reshape[i][j*2] = xmal_reshape[i][j]
            # for j in range(int(800 / period)):
            for j in range(int(80 / period)):
                wr[i][j * period + (period - 1)] = xtest_mal_reshape[i][j]

        # w_reshape = np.reshape(w, (w.shape[0], 128))
        # w_reshape = np.reshape(w, (w.shape[0], 800))
        # w_reshape = np.reshape(wr, (w.shape[0], 1, 40, 20))
        w_reshape = np.reshape(wr, (w.shape[0], 1, 10, 8))

        # xtest_mal_noise_reg = np.ones(xtest_mal_reshape_noise.shape) * (w_reshape > 0.5)
        # test_TRR = gan.blackbox_detector.score(xtest_mal_noise_reg, ytest_mal)
        # test_TRR = gan.blackbox_detector.score(xtest_mal_noise, ytest_mal)
        test_TRR = gan.discriminator.evaluate(w_reshape, ytest_mal)
        Test_TRR.append(test_TRR)

        if (test_TRR < min_test_TRR):
            min_test_TRR = test_TRR
            min_test_epoch = epoch + 1

            # store fake test data
            # xtest_int = (xtest_mal_reshape_noise * 125000.0 + 125000.0).astype(np.int)
            # xtest_int = (wr * 125000.0 + 125000.0).astype(np.int)
            xtest_int = (wr * 35000.0 + 35000.0).astype(np.int)
            # print(xtest_int[0])
            # f = open("fake_unigram_1000.txt", "w")
            # f = open("fake_train_mqtt.csv", "w")
            # f = open("fake_train_mqtt_"+str(period)+".csv", "w")
            f = open("fake_train_mqtt_" + str(period) + ".csv", "a")

            for j in range(xtest_int.shape[0]):
                line = ""
                # for i in range(800):
                for i in range(80):
                    # print(noise_int[i])
                    # if (i != 799):
                    if (i != 79):
                        line += str(xtest_int[j][i]) + ","
                    else:
                        line += str(xtest_int[j][i]) + "\n"
                f.write(line)
            f.close()

        if ((epoch + 1) % 10 == 0):
            print("\ntest_TRR=", test_TRR)
            print("min_test_TRR=", min_test_TRR)
            print("min_test_epoch=", min_test_epoch)

        if ((epoch + 1) % 10 == 0):
            # print("Original_train_TRR=", Original_Train_TRR)
            # print("Original_test_TRR=", Original_Test_TRR)
            et = time.time()
            print("Elapsed time=", (et - st))

    ###############################################################################
    # Plot TRR
    plt.figure()
    plt.plot(range(epochs), Train_TRR, c='r', label='Training Set', linewidth=2)
    plt.plot(range(epochs), Test_TRR, c='g', linestyle='--', label='Validation Set', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()

    # gan.generator.save_weights(output_fold + '/' + 'generator', True)
    # gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)

    # np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    # np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)


def f1_score(y_true, y_pred):
    tp = sum(y_true * y_pred)
    tp_fp = sum(y_pred)
    tp_fn = sum(y_true)

    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = len(y_true) - tp - fp - fn

    precision = tp / tp_fp
    recall = tp / tp_fn
    f1_score = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)

    print('\nTP=', tp)
    print('FP=', fp)
    print('FN=', fn)
    print('TN=', tn)

    print('precision=', precision)
    print('recall=', recall)
    print('fpr=', fpr)


################################
# GAN 예제 실행하기
################################
import argparse


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the networks')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the networks')

    # parser.add_argument('--epochs', type=int, default=10000, help='Epochs for the networks')
    # parser.add_argument('--epochs', type=int, default=1000, help='Epochs for the networks')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for the networks')
    # parser.add_argument('--epochs', type=int, default=10, help='Epochs for the networks')
    # parser.add_argument('--epochs', type=int, default=1, help='Epochs for the networks')

    parser.add_argument('--output_fold', type=str, default='GAN_OUT',
                        help='Output fold to save the results')

    # parser.add_argument('--input_dim', type=int, default=10, help='Input dimension for the generator.')
    # parser.add_argument('--input_dim', type=int, default=128, help='Input dimension for the generator.')
    # parser.add_argument('--input_dim', type=int, default=800, help='Input dimension for the generator.')
    parser.add_argument('--input_dim', type=int, default=80, help='Input dimension for the generator.')

    parser.add_argument('--n_train', type=int, default=32,
                        help='The number of training data.')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()