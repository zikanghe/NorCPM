# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:12:09 2023

@author: Zikang
"""

# load toolbox
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, BatchNormalization, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
from sklearn.utils import shuffle


# file_name = os.listdir('../data/grid/analysisforecast.nc')


nf = nc.Dataset('../data/analysisforecast.nc', 'r')
lon = nf.variables['plon'][:].data
lat = nf.variables['plat'][:].data

aicec = nf.variables['aicen'][:].data
ficec = nf.variables['ficen'][:].data

aicet = nf.variables['aicet'][:].data
ficet = nf.variables['ficet'][:].data

aT = nf.variables['atemp'][:].data
fT = nf.variables['ftemp'][:].data

aS = nf.variables['asalt'][:].data
fS = nf.variables['fsalt'][:].data

nf = nc.Dataset('../data/Region.nc', 'r')
region = nf.variables['region'][:].data

## Define the input data
X = np.stack((ficec, ficet, fT, fS, np.broadcast_to(lat, ficec.shape)), axis=-1)

## Define the output data
y = np.stack((aicec - ficec, aicet - ficet), axis=-1)

## Define the training
s_y = 2014  # Starting year of the training
e_y = 2019  # End year of the training (not included in the training)
n_train = (e_y - s_y) * 12  # Number of training samples
start_indx = (s_y - 2002) * 12
end_indx = start_indx + n_train
xx_train, yy_train = X[start_indx:end_indx], y[start_indx:end_indx]

## Define the validation
n_valyear = 1  # number of year used for validation
n_val = (e_y - 2002) * 12 * n_valyear  # nmuber of validation samples
start_indx_val = (e_y - 2002) * 12
end_indx_val = start_indx_val + n_val
xx_test, yy_test = X[start_indx_val:end_indx_val], y[start_indx_val:end_indx_val]

if False:  # Old code you can remove it
    aicec_train = np.zeros((384, 320, i_train))
    ficec_train = np.zeros((384, 320, i_train))
    aicet_train = np.zeros((384, 320, i_train))
    ficet_train = np.zeros((384, 320, i_train))

    aT_train = np.zeros((384, 320, i_train))
    fT_train = np.zeros((384, 320, i_train))
    aS_train = np.zeros((384, 320, i_train))
    fS_train = np.zeros((384, 320, i_train))
    m_train = np.zeros((i_train))

    for i in range(i_train):
        ind = (s_y - 2002) * 12 + i

        aicec_train[:, :, i] = aicec[:, :, ind]
        ficec_train[:, :, i] = ficec[:, :, ind]

        aicet_train[:, :, i] = aicet[:, :, ind]
        ficet_train[:, :, i] = ficet[:, :, ind]

        aT_train[:, :, i] = aT[:, :, ind]
        fT_train[:, :, i] = fT[:, :, ind]

        aS_train[:, :, i] = aS[:, :, ind]
        fS_train[:, :, i] = fS[:, :, ind]
        print(i, ind)

    i_t = 12
    aicec_test = np.zeros((384, 320, i_t))
    ficec_test = np.zeros((384, 320, i_t))
    aicet_test = np.zeros((384, 320, i_t))
    ficet_test = np.zeros((384, 320, i_t))
    aT_test = np.zeros((384, 320, i_t))
    fT_test = np.zeros((384, 320, i_t))
    aS_test = np.zeros((384, 320, i_t))
    fS_test = np.zeros((384, 320, i_t))

    # n_fl = (e_y-2002)*12
    n_fl = (e_y - 2002) * 12
    for i in range(n_fl, n_fl + i_t):
        ind = (e_y - 2002) * 12 + i
        aicec_test[:, :, i - n_fl] = aicec[:, :, i]
        ficec_test[:, :, i - n_fl] = ficec[:, :, i]

        aicet_test[:, :, i - n_fl] = aicet[:, :, i]
        ficet_test[:, :, i - n_fl] = ficet[:, :, i]

        aT_test[:, :, i - n_fl] = aT[:, :, i]
        fT_test[:, :, i - n_fl] = fT[:, :, i]

        aS_test[:, :, i - n_fl] = aS[:, :, i]
        fS_test[:, :, i - n_fl] = fS[:, :, i]
        print(i)

    d = np.where(aicec == 1)
    aicec[aicec >= 1] = 1

    sst = fT[:, :, 0]

    for ind_r in range(1, 2):
        id1 = np.where(np.logical_and(~np.isnan(sst), region > 0))
        le = len(id1[0])

        xx_train = np.zeros((n_fl * le, 6))
        yy_train = np.zeros((n_fl * le, 2))

        for i in range(le):
            xx_train[i * n_fl:(i + 1) * n_fl, 0] = fT[id1[0][i], id1[1][i], 0:n_fl]
            xx_train[i * n_fl:(i + 1) * n_fl, 1] = fS[id1[0][i], id1[1][i], 0:n_fl]
            xx_train[i * n_fl:(i + 1) * n_fl, 2] = ficec[id1[0][i], id1[1][i], 0:n_fl]
            xx_train[i * n_fl:(i + 1) * n_fl, 3] = ficet[id1[0][i], id1[1][i], 0:n_fl]
            xx_train[i * n_fl:(i + 1) * n_fl, 4] = lon[id1[0][i], id1[1][i]] * np.ones((n_fl))
            xx_train[i * n_fl:(i + 1) * n_fl, 5] = lat[id1[0][i], id1[1][i]] * np.ones((n_fl))

            yy_train[i * n_fl:(i + 1) * n_fl, 0] = aicec[id1[0][i], id1[1][i], 0:n_fl] - ficec[id1[0][i], id1[1][i],
                                                                                         0:n_fl]
            yy_train[i * n_fl:(i + 1) * n_fl, 1] = aicet[id1[0][i], id1[1][i], 0:n_fl] - ficet[id1[0][i], id1[1][i],
                                                                                         0:n_fl]

        id1 = np.where(np.logical_and(~np.isnan(sst), region == ind_r))
        le = len(id1[0])
        xx_test = np.zeros((i_t * le, 6))
        yy_test = np.zeros((i_t * le, 2))
        for i in range(le):
            xx_test[i * i_t:(i + 1) * i_t, 0] = fT_test[id1[0][i], id1[1][i], :]
            xx_test[i * i_t:(i + 1) * i_t, 1] = fS_test[id1[0][i], id1[1][i], :]
            xx_test[i * i_t:(i + 1) * i_t, 2] = ficec_test[id1[0][i], id1[1][i], :]
            xx_test[i * i_t:(i + 1) * i_t, 3] = ficet_test[id1[0][i], id1[1][i], :]
            xx_test[i * i_t:(i + 1) * i_t, 4] = lon[id1[0][i], id1[1][i]] * np.ones((i_t))
            xx_test[i * i_t:(i + 1) * i_t, 5] = lat[id1[0][i], id1[1][i]] * np.ones((i_t))

            yy_test[i * i_t:(i + 1) * i_t, 0] = aicec_test[id1[0][i], id1[1][i], :] - ficec_test[id1[0][i], id1[1][i],
                                                                                      :]
            yy_test[i * i_t:(i + 1) * i_t, 1] = aicet_test[id1[0][i], id1[1][i], :] - ficet_test[id1[0][i], id1[1][i],
                                                                                      :]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

scaler_x.fit(xx_train)
scaler_y.fit(yy_train)

archi = [(100, 'relu'), (50, 'relu')]
reg = 0.0001
batchlayer = {0}
optimizer = 'RMSprop'
batch_size = 512

xx_train, yy_train = shuffle(xx_train, yy_train)
xx_test, yy_test = shuffle(xx_test, yy_test)


def buildmodel_dense(archi, m=4, reg=1e-4, batchlayer={0, 1}):
    inputs = Input(shape=(m,))
    if 0 in batchlayer:
        x = BatchNormalization()(inputs)
    else:
        x = inputs
    for i, (nneur, activ) in enumerate(archi):
        if i + 1 in batchlayer:
            x = BatchNormalization()(x)
        x = Dense(nneur, activation=activ)(x)
    output = Dense(2, activation='linear', kernel_regularizer=regularizers.l2(reg))(x)
    return Model(inputs, output)


kw_modelargs = {
    'archi': archi,
    'm': xx_train.shape[1],
    'reg': reg,
    'batchlayer': batchlayer
}
model = buildmodel_dense(**kw_modelargs)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=50,
    verbose=0,
    mode="auto",
    baseline=None,
    # restore_best_weights=True,
)

opt = tf.keras.optimizers.Adam(learning_rate=0.0015)
model.compile(loss='mse', optimizer=opt)
history11 = model.fit(scaler_x.transform(xx_train), scaler_y.transform(yy_train),
                      epochs=100, batch_size=batch_size, callbacks=[callback],
                      validation_data=(scaler_x.transform(xx_test), scaler_y.transform(yy_test)))
fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogy(history11.history['val_loss'], label='exp1 val_loss', color='gray')
ax.semilogy(history11.history['loss'], label='exp1 loss', color='r')

ax.legend()
plt.title('learning Region ice error Region' + str(ind_r), fontsize=20)
plt.savefig('../0327/' + str(e_y) + '/learning icec Region' + str(ind_r) + '.png')

import joblib

model.save_weights('../0327/' + str(e_y) + '/weight.h5')
joblib.dump(scaler_x, '../0327/' + str(e_y) + '/scalerx.pkl')
joblib.dump(scaler_y, '../0327/' + str(e_y) + '/scalery.pkl')
