from keras.layers import Input, Dense, Activation
from keras.layers import Maximum, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class mqttGAN():
    def __init__(self, filename='D:/workspace/GAN/swGAN/data/process_mqtt_data.npz'):
        pass
    
    '''
    def GENERATOR(self):

        example = Input(shape=(self.apifeature_dims,))
        noise = Input(shape=(self.z_dims,))
        x = Concatenate(axis=1)([example, noise])
        for dim in self.generator_layers[1:]:
            x = Dense(dim)(x)
        x = Activation(activation='sigmoid')(x)
        x = Maximum()([example, x])
        generator = Model([example, noise], x, name='generator')
        generator.summary()
        return generator
    
    def build_detector(self):
        pass     
    '''

    def load_data(self):
        data = np.load(self.filename)
        xmal, ymal, xben, yben = data['x_mal_processed'], data['y_mal'], data['x_normal_processed'], data['y_normal']
        print('sw--success def data_load!')
        return (xmal, ymal), (xben, yben)
                
    
    
    def train(self, epochs, batch_size=32, is_first=1):

        # Load and Split the dataset
        (xmal, ymal), (xben, yben) = self.load_data()
        xtrain_mal, xtest_mal, ytrain_mal, ytest_mal = train_test_split(xmal, ymal, test_size=0.20)
        xtrain_ben, xtest_ben, ytrain_ben, ytest_ben = train_test_split(xben, yben, test_size=0.20)
        if self.same_train_data:
            bl_xtrain_mal, bl_ytrain_mal, bl_xtrain_ben, bl_ytrain_ben = xtrain_mal, ytrain_mal, xtrain_ben, ytrain_ben
        else:
            xtrain_mal, bl_xtrain_mal, ytrain_mal, bl_ytrain_mal = train_test_split(xtrain_mal, ytrain_mal, test_size=0.50)
            xtrain_ben, bl_xtrain_ben, ytrain_ben, bl_ytrain_ben = train_test_split(xtrain_ben, ytrain_ben, test_size=0.50)

        # if is_first is Ture, Train the blackbox_detctor
        if is_first:
            self.blackbox_detector.fit(np.concatenate([xmal, xben]),
                                       np.concatenate([ymal, yben]))

        ytrain_ben_blackbox = self.blackbox_detector.predict(bl_xtrain_ben)
        Original_Train_TPR = self.blackbox_detector.score(bl_xtrain_mal, bl_ytrain_mal)
        Original_Test_TPR = self.blackbox_detector.score(xtest_mal, ytest_mal)
        Train_TPR, Test_TPR = [Original_Train_TPR], [Original_Test_TPR]
        best_TPR = 1.0
        for epoch in range(epochs):

            for step in range(xtrain_mal.shape[0] // batch_size):
                # ---------------------
                #  Train substitute_detector
                # ---------------------

                # Select a random batch of malware examples
                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))
                idx = np.random.randint(0, xmal_batch.shape[0], batch_size)
                xben_batch = xtrain_ben[idx]
                yben_batch = ytrain_ben_blackbox[idx]

                # Generate a batch of new malware examples
                gen_examples = self.generator.predict([xmal_batch, noise])
                ymal_batch = self.blackbox_detector.predict(np.ones(gen_examples.shape)*(gen_examples > 0.5))

                # Train the substitute_detector
                d_loss_real = self.substitute_detector.train_on_batch(gen_examples, ymal_batch)
                d_loss_fake = self.substitute_detector.train_on_batch(xben_batch, yben_batch)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                idx = np.random.randint(0, xtrain_mal.shape[0], batch_size)
                xmal_batch = xtrain_mal[idx]
                noise = np.random.uniform(0, 1, (batch_size, self.z_dims))

                # Train the generator
                g_loss = self.combined.train_on_batch([xmal_batch, noise], np.zeros((batch_size, 1)))

            # Compute Train TPR
            noise = np.random.uniform(0, 1, (xtrain_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtrain_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytrain_mal)
            Train_TPR.append(TPR)

            # Compute Test TPR
            noise = np.random.uniform(0, 1, (xtest_mal.shape[0], self.z_dims))
            gen_examples = self.generator.predict([xtest_mal, noise])
            TPR = self.blackbox_detector.score(np.ones(gen_examples.shape) * (gen_examples > 0.5), ytest_mal)
            Test_TPR.append(TPR)

            # Save best model
            if TPR < best_TPR:
                self.combined.save_weights('saves/malgan.h5')
                best_TPR = TPR

            # Plot the progress
            if is_first:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        flag = ['DiffTrainData', 'SameTrainData']
        print('\n\n---{0} {1}'.format(self.blackbox, flag[self.same_train_data]))
        print('\nOriginal_Train_TPR: {0}, Adver_Train_TPR: {1}'.format(Original_Train_TPR, Train_TPR[-1]))
        print('\nOriginal_Test_TPR: {0}, Adver_Test_TPR: {1}'.format(Original_Test_TPR, Test_TPR[-1]))

        # Plot TPR
        plt.figure()
        plt.plot(range(len(Train_TPR)), Train_TPR, c='r', label='Training Set', linewidth=2)
        plt.plot(range(len(Test_TPR)), Test_TPR, c='g', linestyle='--', label='Validation Set', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig('saves/Epoch_TPR({0}, {1}).png'.format(self.blackbox, flag[self.same_train_data]))
        plt.show()
    
if __name__ == '__main__':
    start = mqttGAN
#    malgan = MalGAN(blackbox='MLP')
#    malgan.train(epochs=500, batch_size=64)
#    malgan.retrain_blackbox_detector()
#    malgan.train(epochs=100, batch_size=64, is_first=False)
    print('sw--success main!')