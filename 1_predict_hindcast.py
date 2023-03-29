# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:34:44 2023

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
nf = nc.Dataset('../data/ocn_grid.nc','r')
lon = nf.variables['plon'][:].data
lat = nf.variables['plat'][:].data



nf = nc.Dataset('../data/analysisforecast.nc','r')
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
d = np.where(aicec==1)
aicec[aicec>=1]=1

hicec = np.copy(ficec[:,:,96:216])
hicet = np.copy(ficet[:,:,96:216])


i_t = 12
i_v = 12
#%%
e_y =2016
aicec_test = np.zeros((384,320,i_t))
ficec_test = np.zeros((384,320,i_t))
aicet_test = np.zeros((384,320,i_t))
ficet_test = np.zeros((384,320,i_t))
aT_test = np.zeros((384,320,i_t))
fT_test = np.zeros((384,320,i_t))
aS_test = np.zeros((384,320,i_t))
fS_test = np.zeros((384,320,i_t))
#n_fl = (e_y-2002)*12
n_fl = (e_y-2002)*12
for i in range(n_fl,n_fl+i_t):


    aicec_test[:,:,i-n_fl] = aicec[:,:,i]
    ficec_test[:,:,i-n_fl] = ficec[:,:,i]

    aicet_test[:,:,i-n_fl] = aicet[:,:,i]
    ficet_test[:,:,i-n_fl] = ficet[:,:,i]

    aT_test[:,:,i-n_fl] = aT[:,:,i]
    fT_test[:,:,i-n_fl] = fT[:,:,i]

    aS_test[:,:,i-n_fl] = aS[:,:,i]
    fS_test[:,:,i-n_fl] = fS[:,:,i]
    print(i)

 




nf = nc.Dataset('../data/Region.nc','r')
region = nf.variables['region'][:].data
sst = fT[:,:,0]

id1 = np.where(np.logical_and(~np.isnan(sst),region>0))
le = len(id1[0])

xx_val = np.zeros((120*le,6))
yy_val = np.zeros((120*le,4))

for i in range(le):
	
    		xx_val[i*i_v:(i+1)*i_v,0] = fT_test[id1[0][i],id1[1][i],:]
    		xx_val[i*i_v:(i+1)*i_v,1] = fS_test[id1[0][i],id1[1][i],:]
    		xx_val[i*i_v:(i+1)*i_v,2] = ficec_test[id1[0][i],id1[1][i],:]
    		xx_val[i*i_v:(i+1)*i_v,3] = ficet_test[id1[0][i],id1[1][i],:]
    		xx_val[i*i_v:(i+1)*i_v,4] = lon[id1[0][i],id1[1][i]]*np.ones((i_v))
    		xx_val[i*i_v:(i+1)*i_v,5] = lat[id1[0][i],id1[1][i]]*np.ones((i_v))
        


import joblib
archi = [(100, 'relu'), (50, 'relu')]
reg = 0.0001
batchlayer = {0}
batch_size = 512
def buildmodel_dense(archi, m=4, reg=1e-4, batchlayer={0,1}):

    	    inputs = Input(shape=(m,))
    	    if 0 in batchlayer:
    	        x = BatchNormalization()(inputs)
    	    else:
    	        x = inputs
    	    for i, (nneur, activ) in enumerate(archi):
    	        if i+1 in batchlayer:
    	            x = BatchNormalization()(x)
    	        x = Dense(nneur, activation=activ)(x)
    	    output = Dense(2, activation='linear', kernel_regularizer=regularizers.l2(reg))(x)
    	    return Model(inputs, output)

kw_modelargs = {
    	        'archi': archi,
    	        'm':xx_val.shape[1],
    	        'reg': reg,
    	        'batchlayer':batchlayer
        }
model = buildmodel_dense(**kw_modelargs)
def load_nn_ice(s_y):
    	    model_nn = buildmodel_dense(**kw_modelargs)
    	    model_nn.load_weights('../0327/'+str(e_y)+'/weight.h5')
    	    scaler_x = joblib.load('../0327/'+str(e_y)+'/scalerx.pkl')
    	    scaler_y = joblib.load('../0327/'+str(e_y)+'/scalery.pkl')
    	    return model_nn,scaler_x,scaler_y

model_nn, scaler_x, scaler_y = load_nn_ice(e_y)
yy_pval=scaler_y.inverse_transform(model_nn.predict(scaler_x.transform(xx_val)))


ind = (e_y-2010)*12

for i in range(le):
        
    		hicec[id1[0][i],id1[1][i],ind:ind+i_v] = ficec_test[id1[0][i],id1[1][i],:]+yy_pval[i*i_v:(i+1)*i_v,0]
    		hicet[id1[0][i],id1[1][i],ind:ind+i_v] = ficet_test[id1[0][i],id1[1][i],:]+yy_pval[i*i_v:(i+1)*i_v,1]
    		
#%%
# Build Hybrid 
import numpy as np
import netCDF4 as nc
nf = nc.Dataset('../data/ocn_grid.nc','r')
lon = nf.variables['plon'][:].data
lat = nf.variables['plat'][:].data
depth = nf.variables['pdepth'][:].data

import netCDF4 as nc
import numpy as np
f_w = nc.Dataset('hybrid_lead_hindcast.nc','w',format = 'NETCDF4')


ny = 320
nx = 384
nz = 120
f_w.createDimension('lat',ny)
f_w.createDimension('lon',nx)
f_w.createDimension('time',nz)



#创建变量
latitudes = f_w.createVariable('plat', np.float32, ('lon','lat'))
longitudes = f_w.createVariable('plon', np.float32,('lon','lat'))
time = f_w.createVariable('time', np.float32, ('time'))


#设置变量值
latitudes[:,:] = lat
longitudes[:,:] = lon
time[:] = np.zeros(120)



f_w.createVariable( 'hicen', np.float32, ('lon','lat','time'))
f_w.variables['hicen'][:] = hicec 

f_w.createVariable( 'hicet', np.float32, ('lon','lat','time'))
f_w.variables['hicet'][:] = hicet 



#关闭文件
f_w.close()