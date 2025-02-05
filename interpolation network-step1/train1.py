from model import build_unet
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import h5py
import tensorflow as tf
import keras.backend as K
from keras import Model
import matplotlib.pyplot as plt
import time as t
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

m = 500
X_train = np.zeros((m, 96, 96, 2))
Y_train = np.zeros((m, 96, 96, 5))
datax = h5py.File(r'.\实验数据驱动\cominput2_8.mat')
data = datax.keys()
data = datax['cominput']
data = np.transpose(data,(3,2,1,0))
# datay = scio.loadmat(r'C:\Users\zsy\Documents\\3D-火焰训练\训练\comoutput.mat')
epochs = 3000

datay = h5py.File(r'.\实验数据驱动\cominput3-7.mat')
datayy = datay.keys()
datayy = datay['cominput']
datayy = np.transpose(datayy,(3,2,1,0))

for i in range(m):
    X_train[i] = data[i] / np.max(data[i])
    datayy = datayy / np.max(datayy)
    # m = np.max(datay)
    Y_train[i] = np.array(datayy[i])

sim_X_train = np.zeros((4096, 96, 96, 2))
sim_Y_train = np.zeros((4096, 96, 96, 5))
simdatax = h5py.File(r'E:\zhu\2022_9_17_925模拟场484896\cominputb2_8.mat')
simdata = simdatax.keys()
simdata = simdatax['cominput']
simdata = np.transpose(simdata,(3,2,1,0))
# datay = scio.loadmat(r'C:\Users\zsy\Documents\\3D-火焰训练\训练\comoutput.mat')
epochs = 3000

simdatay = h5py.File(r'E:\zhu\2022_9_17_925模拟场484896\cominputb3_7.mat')
simdatayy = simdatay.keys()
simdatayy = simdatay['cominput']
simdatayy = np.transpose(simdatayy,(3,2,1,0))

for i in range(m):
    sim_X_train[i] = simdata[i] / np.max(simdata[i])
    simdatayy = simdatayy / np.max(simdatayy)
    # m = np.max(datay)
    sim_Y_train[i] = np.array(simdatayy[i])
X_train = np.append(X_train,sim_X_train,axis=0)
Y_train = np.append(Y_train,sim_Y_train,axis=0)
X_train, X_test, Y_train, Y_test =train_test_split(X_train,Y_train,test_size=0.1, random_state=1)

X_train = np.reshape(X_train, (4136, 96, 96, 2))
X_test = np.reshape(X_test, (460, 96, 96, 2))
Y_train = np.reshape(Y_train, (4136, 96, 96, 5))
Y_test = np.reshape(Y_test, (460, 96, 96, 5))
TTest = X_test[2,:,:,:].reshape(1,96,96,2)
np.save('X_test.npy',TTest)
print(X_train.shape)
print(Y_train.shape)

def ssim_fun(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)

def plotqieimg(truth_value, value, epoch=0):
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    img1_truth = truth_value[23, :, :]
    # img1_truth = np.reshape(img1_truth,(48,48))
    ax[0, 0].imshow(img1_truth, cmap='hot')
    ax[0, 0].set_axis_off(), ax[0, 0].set_title('truth x=24', fontsize=20)
    img2_truth = truth_value[:, 23, :]
    ax[0, 1].imshow(img2_truth, cmap='hot')
    ax[0, 1].set_axis_off(), ax[0, 1].set_title('truth y=24', fontsize=20)
    img3_truth = truth_value[:, :, 2]
    ax[0, 2].imshow(img3_truth, cmap='hot')
    ax[0, 2].set_axis_off(), ax[0, 2].set_title('truth z=3', fontsize=20)
    img4_truth = truth_value[:, :, 23]
    ax[0, 3].imshow(img4_truth, cmap='hot')
    ax[0, 3].set_axis_off(), ax[0, 3].set_title('truth z=24', fontsize=20)
    img1 = value[0,23, :, :]
    ax[1, 0].imshow(img1, cmap='hot')
    ax[1, 0].set_axis_off(), ax[1, 0].set_title('sim x=24', fontsize=20)
    img2 = value[0,:, 23, :]
    ax[1, 1].imshow(img2, cmap='hot')
    ax[1, 1].set_axis_off(), ax[1, 1].set_title('sim y=24', fontsize=20)
    img3 = value[0,:, :, 2]
    ax[1, 2].imshow(img3, cmap='hot')
    ax[1, 2].set_axis_off(), ax[1, 2].set_title('sim z=3', fontsize=20)
    img4 = value[0,:, :, 23]
    ax[1, 3].imshow(img4, cmap='hot')
    ax[1, 3].set_axis_off(), ax[1, 3].set_title('sim z=24', fontsize=20)
    fig.savefig('./images/res', bbox_inches='tight', pad_inches=0.1)
    plt.close()

model = build_unet()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mse', optimizer=adam,metrics=ssim_fun)
model.summary()

model.fit(X_train, Y_train, validation_split=0.2,  batch_size=32, epochs=3000, shuffle=True)
model.save('flame.h5')
model.save_weights('weights.h5')
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
cosine_proximity = model.history.history['ssim_fun']
val_cosine_proximity = model.history.history['val_ssim_fun']
scio.savemat('loss.mat', {"loss": loss})
scio.savemat('val_loss.mat', {"val_loss": val_loss})
scio.savemat('ssim.mat', {"cosine_proximity": cosine_proximity})
scio.savemat('val_ssim.mat', {"val_cosine_proximity": val_cosine_proximity})
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].plot(loss)
ax[1].plot(cosine_proximity)
plt.savefig('./images/loss_cosine', bbox_inches='tight', pad_inches=0.1)
plt.close()
# result = model.predict(X_train)
#
# layer_name = 'output'
# layer_output = model.get_layer(layer_name).output
# layer_input = model.input
# final = Model(inputs=layer_input, outputs=layer_output)
# result = final.predict(X_train)




# scores = model.evaluate(X_test, Y_test)
# print("Baseline Error: %.2f%%" %(100-scores[1] *100))
