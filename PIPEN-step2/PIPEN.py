# -*- coding: utf-8 -*-

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

from PIL import Image
from scipy import signal
# from get_train import Phy_Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Convolution2D, LeakyReLU, BatchNormalization, UpSampling2D, Dropout, Activation, Flatten, \
    Dense, Lambda, Reshape, concatenate, Convolution3D,UpSampling3D,Add
from skimage.metrics import structural_similarity as SSIM
import FileUnit   as FL
import h5py
import time as t
import scipy.io as scio
from sklearn.metrics import mean_squared_error
# from skimage.measure import compare_mse
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import gc
import math
import scipy.sparse as sparse

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.enable_eager_execution()

class U_Net_Phy():
    def __init__(self, name='52'):
        self.depth = 65535
        self.channels = 12
        self.height = 96
        self.width = 96
        self.shape = (self.height, self.width, self.channels)
        self.shape1 = (self.height, self.width)

        # 编译参数及优化器
        self.lamda = 0.0002
        self.simple_num = 2000
        self.batch_size = 1
        optimizer = Adam(learning_rate=0.005)

        # 获取训练数据
        # self.x_train, self.x_label = self.load_data()

        # U-net
        self.unet = self.build_unet()
        self.unet.compile(loss=self.loss_fun, optimizer=optimizer)
        self.unet.run_eagerly = True
        self.unet.summary()



    def build_unet(self):
        """
        Create the U-Net Generator using the hyperparameter values defined below
        """
        kernel_size = 4
        strides = 2
        leakyrelu_alpha = 0.2
        upsampling_size = (2, 2)
        dropout = 0.5
        output_channels = 96
        output = (1, 48, 48)
        input_shape = (96, 96, 7)

        input_layer = Input(shape=input_shape)

        # Encoder Network
        enc1 = Convolution2D(filters=16,kernel_size=kernel_size,strides=(2,2),padding='same',name='input12')(input_layer)
        # 1st Convolutional block in the encoder network
        encoder1 = Convolution2D(filters=32, kernel_size=kernel_size, padding='same',
                                 strides=strides,name='enc1')(enc1)
        encoder1 = LeakyReLU(alpha=leakyrelu_alpha,name='enl1')(encoder1)
        # encoder1 = Activation('relu')(encoder1)

        # 2nd Convolutional block in the encoder network
        encoder2 = Convolution2D(filters=64, kernel_size=kernel_size, padding='same',
                                 strides=strides,name='enc2')(encoder1)
        encoder2 = BatchNormalization(name='enb2')(encoder2)
        encoder2 = LeakyReLU(alpha=leakyrelu_alpha,name='enl2')(encoder2)
        # encoder2 = Activation('relu')(encoder2)

        # 3rd Convolutional block in the encoder network
        encoder3 = Convolution2D(filters=128, kernel_size=kernel_size, padding='same',
                                 strides=strides,name='enc3')(encoder2)
        encoder3 = BatchNormalization(name='enb3')(encoder3)
        encoder3 = LeakyReLU(alpha=leakyrelu_alpha,name='enl3')(encoder3)
        # encoder3 = Activation('relu')(encoder3)

        # 4th Convolutional block in the encoder network
        encoder4 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',
                                 strides=strides,name='enc4')(encoder3)
        encoder4 = BatchNormalization(name='enb4')(encoder4)
        encoder4 = LeakyReLU(alpha=leakyrelu_alpha,name='enl4')(encoder4)
        # encoder4 = Activation('relu')(encoder4)


        # 1st Upsampling Convolutional Block in the decoder network
        decoder1 = UpSampling2D(size=upsampling_size,name='deu1')(encoder4)
        decoder1 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',name='dec1')(decoder1)
        decoder1 = BatchNormalization(name='deb1')(decoder1)
        decoder1 = Dropout(dropout,name='ded1')(decoder1)
        decoder1 = concatenate([decoder1, encoder3], axis=3,name='decc1')
        decoder1 = LeakyReLU(alpha=leakyrelu_alpha,name='del1')(decoder1)

        # 2nd Upsampling Convolutional block in the decoder network
        decoder2 = UpSampling2D(size=upsampling_size,name='deu2')(decoder1)
        decoder2 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',name='dec2')(decoder2)
        decoder2 = BatchNormalization(name='deb2')(decoder2)
        decoder2 = Dropout(dropout,name='ded2')(decoder2)
        decoder2 = concatenate([decoder2, encoder2], axis=3,name='decc2')
        decoder2 = LeakyReLU(alpha=leakyrelu_alpha,name='del2')(decoder2)

        # 3rd Upsampling Convolutional block in the decoder network
        decoder3 = UpSampling2D(size=upsampling_size,name='deu3')(decoder2)
        decoder3 = Convolution2D(filters=128, kernel_size=kernel_size, padding='same',name='dec3')(decoder3)
        decoder3 = BatchNormalization(name='deb3')(decoder3)
        decoder3 = Dropout(dropout,name='ded3')(decoder3)
        decoder3 = concatenate([decoder3, encoder1], axis=3,name='decc3')
        decoder3 = LeakyReLU(alpha=leakyrelu_alpha,name='del3')(decoder3)

        # 4th Upsampling Convolutional block in the decoder network
        decoder4 = UpSampling2D(size=upsampling_size,name='deu4')(decoder3)
        decoder4 = Convolution2D(filters=output_channels, kernel_size=kernel_size, padding='same',name='dec4')(decoder4)
        decoder4 = BatchNormalization(name='deb4')(decoder4)

        decoder5 = Activation('relu',name='output')(decoder4)

        model = Model(inputs=[input_layer], outputs=[decoder5])
        return model


    def loss_fun(self, y_true, y_pred):
        tic = t.time()
        y_pre = tf.reshape(y_pred,(48,48,96))
        # y_pre = y_pre/ tf.reduce_max(y_pre)
        reshapetime = t.time()
        print('reshape time = %f' % (reshapetime - tic))
        # y_pred = tf.transpose(y_pred,(2,0,1))
        y_pred = self.projection(y_pre)
        y_train = y_pred[0, :, :, 1:8]
        toc = t.time()
        print('projection time = %f' %(toc-tic))
        TV = self.Tv(y_pre)
        MSE = K.mean(K.square(y_train - y_true), axis=-1)+10*TV
        print('mse: %f, Tv: %f' % (K.mean(MSE).numpy(), TV.numpy()))
        return MSE  # 返回mse

    def Tv(self, value):
        tv_h = K.sum(K.square(value[:, 1:, :] - value[:, :-1, :]))
        tv_w = K.sum(K.square(value[:, :, 1:] - value[:, :, :-1]))
        tv_c = K.sum(K.square(value[1, :, :] - value[:-1, :, :]))
        return (K.sqrt(tv_c + tv_w + tv_h)) / (48 * 48 * 96)

    def projection(self, value):
        # value = tf.cast(value, dtype=tf.float32)
        tic = t.time()
        with tf.device('/cpu:0'):
            value = tf.transpose(value, (2, 1, 0))
            # value = tf.cast(value,dtype=tf.float32)
            pro_all = tf.constant(np.zeros((1, 96, 96, 0)), dtype=tf.float32)
            # weight_96 = np.zeros((96,1))
            for pn in range(1, 13):
                pro = tf.constant(np.zeros((0, 48)), dtype=tf.float32)
                weights = sparse.load_npz('./matrix/sparse_martix%d.npz' % (pn))
                indices = np.vstack((weights.row, weights.col)).astype(np.int64)
                values = weights.data
                value = tf.reshape(value, [-1, 1])

                sparse_tensor = tf.compat.v1.SparseTensorValue(indices=indices.transpose((1, 0)), values=values,
                                                               dense_shape=[96 * 96, 48 * 48 * 96])

                tensor = tf.compat.v1.sparse_tensor_dense_matmul(sparse_tensor, value)
                pro = tf.reshape(tensor, (96, 96))
                pro_max = tf.reduce_max(pro)
                pro = pro / pro_max
                # pro = tf.reshape(pro,(96,96))
                pro_1 = tf.reshape(pro, (1, 96, 96, 1))
                pro_all = tf.concat([pro_all, pro_1], axis=3)
            toc = t.time()
            print('all projection time=%f' % (toc - tic))
            return pro_all



    def load_data(self):
        X_train = np.zeros((1, 96, 96, 7))

        datax = h5py.File(r'.\671\cominputb10.mat')
        data = datax.keys()
        data = datax['cominput']
        data = np.transpose(data, (3, 2, 1, 0))
        # xx=np.reshape(X_train[0,:,:,1],(96,96))
        # plt.imshow(xx)
        # plt.show()
        X4 = data[0,:,:,0] / np.max(data[0,:,:,0])
        datainter = scio.loadmat('.\\671\\671-inter.mat')['output']
        X_train = datainter
        datay = scio.loadmat(r'.\671\R%d.mat' % (671))
        datay = datay['flame']
        M = np.max(datay)
        datay = datay / np.max(datay)

        Y_train = np.array(datay)

        return X_train, X_train, Y_train,X4

    def psnr2(self, img1, img2):
        mse = np.mean((img1 - img2 ) ** 2)
        if mse < 1e-10:
            return 100
        psnr2 = 20 * math.log10(1 / math.sqrt(mse))
        return psnr2

    def mean2(self,x):
        y = np.sum(x) / np.size(x)
        return y

    def corr2(self,a, b):
        a = a - self.mean2(a)
        b = b - self.mean2(b)

        r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
        return r

    def train(self, epochs=101, batch_size=32, sample_interval=10):
        os.makedirs('./weights', exist_ok=True)
        # 获得数据
        x_train, x_label, truth_value,X4 = self.load_data()  # self.load_data() #

        x_label = x_train
        # x_label = x_label / np.max(x_label)
        total_num = self.simple_num

        if os.path.exists('./weights/epoch_model.h5'):
            self.unet.load_weights('./weights/epoch_model.h5')
        # ''' # train_on_batch的方法训练网络
        loss, ssim, rmse, psnr , corr = [], [] ,[], [],[]
        tmp = 0
        for epoch in range(epochs):
            # 保存权重 迁移学习保证第一个是迁移的结果
            all_tic = t.time()
            if epoch % 1 == 0:

                os.makedirs('./weights/model_2024_9_19_1610/', exist_ok=True)
                self.unet.save_weights('./weights/model_2024_9_19_1610/epoch%d_model.h5' % epoch)
                value = self.unet.predict(x_train)
                # scio.savemat('try1.mat', {'foo': value})
                value = tf.reshape(value,(48,48,96))

                pt1 = t.time()
                img2 = self.projection(value)
                img2_4 = img2[0,:, :, 0].numpy()
                pt2 = t.time()
                print("save projetcion = %f"%(pt2-pt1))
                # x_label = tf.cast(X4,dtype=tf.float32)
                train_ssim = SSIM(X4,img2_4,max_val=1)
                train_rmse = np.sqrt(mean_squared_error(X4,img2_4))
                train_psnr = self.psnr2(X4,img2_4)
                train_corr = self.corr2(X4,img2_4)
                img2 = img2.numpy()
                x_label = x_label
                # mse = mean_squared_error(img2.reshape(1, -1), x_label.reshape(1, -1))
                # print('mse = %f'%(mse))
                tic12 = t.time()
                self.plot4img(x_label,X4,img2,epoch)
                toc12 = t.time()
                print('plot 12 time = %f' %(toc12-tic12))
                # self.plot2img(x_label[0,:,:,0], img2[0,:,:,0], epoch)
                # self.plot2img(x_label[0, :, :, 0],x_label[0, :, :, 1], epoch)
                value = tf.transpose(value,(0,1,2))
                qietic = t.time()
                self.plotqieimg(truth_value,value,epoch)
                qietoc = t.time()
                print('qie time = %f' %(qietoc-qietic))

            tt = t.time()
            train_loss = self.unet.train_on_batch(x_train, x_label)
            all_toc = t.time()
            print('train on batch time = %f'%(all_toc-tt))
            print('epoch time = %f' % (all_toc - all_tic))
            #保存最优的模型
            # if train_ssim > tmp:
            #     tmp = train_ssim
            #     self.unet.save_weights('./weights/best_epoch_weights.h5')
            #     print("I had save the best weights")

            print('schedule: %d/%d' % (epoch, epochs),
                  ' - loss: %f - ssim: %f - rmse: %f - psnr: %f - corr: %f' % (train_loss,train_ssim,train_rmse,train_psnr,train_corr))
            loss.append(train_loss)
            ssim.append(train_ssim)
            rmse.append(train_rmse)
            psnr.append(train_psnr)
            corr.append(train_corr)



        # '''

        ''' # 使用fit训练
        weight_path = './weights/best_model.h5'  # self.weight_path
        callbacks = [ModelCheckpoint(weight_path, verbose=1, save_best_only=True)]
        result = self.unet.fit(x_train, x_label, batch_size=batch_size, epochs=epochs, verbose=2,
                               callbacks=callbacks, validation_data=(x_train, x_label))
        loss, ssim = result.history['loss'], result.history['val_ssim_fun']
        self.unet.save_weights('./weights/best_weights.h5')
        '''
        # self.test2(name='52', epoch=46)
        return loss,ssim,rmse,psnr,corr



    def plotqieimg(self,truth_value,value,epoch=0):
        fig, ax = plt.subplots(2,5,figsize=(24,18))
        img1_truth = truth_value[23,:,:]
        # img1_truth = np.reshape(img1_truth,(48,48))
        ax[0,0].imshow(img1_truth, cmap='gray')
        ax[0,0].set_axis_off(), ax[0,0].set_title('truth x=24', fontsize=20)
        img2_truth = truth_value[:,23,:]
        ax[0,1].imshow(img2_truth, cmap='gray')
        ax[0,1].set_axis_off(), ax[0,1].set_title('truth y=24', fontsize=20)
        img3_truth = truth_value[:,:,0]
        ax[0,2].imshow(img3_truth, cmap='gray')
        ax[0,2].set_axis_off(), ax[0,2].set_title('truth z=21', fontsize=20)
        img4_truth = truth_value[:,:,1]
        ax[0,3].imshow(img4_truth, cmap='gray')
        ax[0,3].set_axis_off(), ax[0,3].set_title('truth z=41', fontsize=20)
        img5_truth = truth_value[:, :, 60]
        ax[0, 4].imshow(img5_truth, cmap='gray')
        ax[0, 4].set_axis_off(), ax[0, 4].set_title('truth z=61', fontsize=20)

        img1 = value[23,:,:]
        ax[1,0].imshow(img1, cmap='gray')
        ax[1,0].set_axis_off(), ax[1,0].set_title('sim x=24', fontsize=20)
        img2 = value[:, 23, :]
        ax[1,1].imshow(img2, cmap='gray')
        ax[1,1].set_axis_off(), ax[1,1].set_title('sim y=24', fontsize=20)
        img3 = value[:, :, 0]
        ax[1,2].imshow(img3, cmap='gray')
        ax[1,2].set_axis_off(), ax[1,2].set_title('sim z=21', fontsize=20)
        img4 = value[:, :, 1]
        ax[1,3].imshow(img4, cmap='gray')
        ax[1,3].set_axis_off(), ax[1,3].set_title('sim z=41', fontsize=20)
        img5 = value[:, :, 60]
        ax[1, 4].imshow(img5, cmap='gray')
        ax[1, 4].set_axis_off(), ax[1, 4].set_title('sim z=61', fontsize=20)
        os.makedirs('./images_2024_9_19_1610/qiepian/', exist_ok=True)
        fig.savefig('./images_2024_9_19_1610/qiepian/epoch%d' % epoch, bbox_inches='tight', pad_inches=0.1)
        plt.close()


    def plot4img(self,img1,X4,img2,epoch=0):
        fig, ax = plt.subplots(4, 4, figsize=(32, 18))
        img1_truth = img1[0, :, :, 0]
        # img1_truth = np.reshape(img1_truth,(48,48))
        ax[0, 0].imshow(img1_truth, cmap='gray')
        ax[0, 0].set_axis_off(), ax[0, 0].set_title('truth 2', fontsize=20)
        img2_truth = img1[0, :, :, 1]
        ax[0, 1].imshow(img2_truth, cmap='gray')
        ax[0, 1].set_axis_off(), ax[0, 1].set_title('truth 3', fontsize=20)
        img3_truth = img1[0, :, :, 2]
        ax[0, 2].imshow(img3_truth, cmap='gray')
        ax[0, 2].set_axis_off(), ax[0, 2].set_title('truth 4', fontsize=20)
        img4_truth = img1[0, :, :, 3]
        ax[0, 3].imshow(img4_truth, cmap='gray')
        ax[0, 3].set_axis_off(), ax[0, 3].set_title('truth 5', fontsize=20)

        img5_truth = img1[0, :, :, 4]
        ax[2, 0].imshow(img5_truth, cmap='gray')
        ax[2, 0].set_axis_off(), ax[2, 0].set_title('truth 6', fontsize=20)
        img6_truth = img1[0, :, :, 5]
        ax[2, 1].imshow(img6_truth, cmap='gray')
        ax[2, 1].set_axis_off(), ax[2, 1].set_title('truth 7', fontsize=20)

        img1_sim = img2[0, :, :, 1]
        ax[1, 0].imshow(img1_sim, cmap='gray')
        ax[1, 0].set_axis_off(), ax[1, 0].set_title('sim 2', fontsize=20)
        img2_sim = img2[0, :, :, 2]
        ax[1, 1].imshow(img2_sim, cmap='gray')
        ax[1, 1].set_axis_off(), ax[1, 1].set_title('sim 3', fontsize=20)
        img3_sim = img2[0, :, :, 3]
        ax[1, 2].imshow(img3_sim, cmap='gray')
        ax[1, 2].set_axis_off(), ax[1, 2].set_title('sim 4', fontsize=20)
        img4_sim = img2[0, :, :, 4]
        ax[1, 3].imshow(img4_sim, cmap='gray')
        ax[1, 3].set_axis_off(), ax[1, 3].set_title('sim 5', fontsize=20)

        img5_sim = img2[0, :, :, 5]
        ax[3, 0].imshow(img5_sim, cmap='gray')
        ax[3, 0].set_axis_off(), ax[3, 0].set_title('sim 6', fontsize=20)
        img6_sim = img2[0, :, :, 6]
        ax[3, 1].imshow(img6_sim, cmap='gray')
        ax[3, 1].set_axis_off(), ax[3, 1].set_title('sim 7', fontsize=20)

        img7_truth = img1[0, :, :, 6]
        # img1_truth = np.reshape(img1_truth,(48,48))
        ax[2, 2].imshow(img7_truth, cmap='gray')
        ax[2, 2].set_axis_off(), ax[2, 2].set_title('truth 8', fontsize=20)

        img12_truth = X4
        ax[2, 3].imshow(img12_truth, cmap='gray')
        ax[2, 3].set_axis_off(), ax[2, 3].set_title('truth 1', fontsize=20)


        img7_sim = img2[0, :, :, 7]
        ax[3, 2].imshow(img7_sim, cmap='gray')
        ax[3, 2].set_axis_off(), ax[3, 2].set_title('sim 8', fontsize=20)

        img12_sim = img2[0, :, :, 0]
        ax[3, 2].imshow(img12_sim, cmap='gray')
        ax[3, 2].set_axis_off(), ax[3, 2].set_title('sim 1', fontsize=20)



        os.makedirs('./images_2024_9_19_1610/3/', exist_ok=True)
        fig.savefig('./images_2024_9_19_1610/3/epoch%d' % epoch, bbox_inches='tight', pad_inches=0.1)
        plt.close()



if __name__ == '__main__':
    # '''# 创建一个保存图片的文件夹
    if not os.path.exists('./images'):
        os.makedirs('./images')
    # '''

    # 运行衍射受限物理模型
    name = '52'
    Phy = U_Net_Phy(name=name)

    ''' 保存训练数据
    # train, label = np.load('data.npy')
    # print(train.shape)
    if not os.path.exists('../x_label1.npy'):
        x_train, x_label1 = Phy.load_data()
        np.save('../x_train.npy', x_train)
        np.save('../x_label1.npy', x_label1)
    '''

    # '''   # 训练模型
    # Phy.unet.load_weights('./weights/best_model%s.h5' % std)
    times = 1
    epoch = 2000
    loss,ssim,rmse,psnr,corr = Phy.train(epochs=epoch, batch_size=1, sample_interval=1)
    # Phy.test1()  # 随机选取三幅图查看结果
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    scio.savemat('loss.mat',{'loss':loss})
    scio.savemat('ssim.mat',{'ssim':ssim})
    scio.savemat('rmse.mat',{'rmse':rmse})
    scio.savemat('psnr.mat', {'psnr': psnr})
    scio.savemat('corr.mat', {'corr': corr})
    ax[0,0].plot(ssim)
    ax[0,1].plot(rmse)
    ax[1, 0].plot(psnr)
    ax[1, 1].plot(corr)
    plt.show()

