import numpy as np
import keras
from keras.layers import Input,Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import pdb
from scipy.io import loadmat

model_save_path = './model/improvement-{epochs:02d}-{loss:.5f}.hdf5'
model_save_path2 ='./model2/improvement-{epochs:02d}-{loss:.5f}.hdf5'

width = 100
height = 100

inputs_path = 'g1.txt'
outputs_path ='g2.txt'
N = 5000
sample_number = 5000

X = loadmat(inputs_path)['g1']
X = np.expand_dims(X,axis=-1)

y = loadmat(outputs_path)['g2']
y = np.expand_dims(y,axis=-1)

'''
def conv_model(block_number=500,filter_size=5,strides=1):
    input_layer = Input(shape=(sample_number,1))
    for i in range(block_number):
        if i ==0:
            x = Conv1D(1,filter_size,padding='same',strides=strides,activation='linear',name='conv'+str(i+1))(input_layer)
        elif i!=block_number-1:
            x = Conv1D(1,filter_size,padding='same',strides=strides,activation='linear',name='conv'+str(i+1))(x)
        else:
            output = Conv1D(1,filter_size,padding='same',strides=strides,activation='linear',name='output')(x)

    model = Model(input_layer,output)
    return model
'''
	
def conv_model():
    input_layer = Input(shape=(sample_number,1))
    conv1 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv1')(input_layer)
    conv2 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv2')(conv1)
    conv3 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv3')(conv2)
    conv4 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv4')(conv3)
    conv5 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv5')(conv4)
    conv6 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv6')(conv5)
    conv7 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv7')(conv6)	
    conv8 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv8')(conv7)	
    conv9 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv9')(conv8)	
    conv10 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv10')(conv9)	
    conv11 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv11')(conv10)	
    conv12 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv12')(conv11)	
    conv13 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv13')(conv12)	
    conv14 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv14')(conv13)	
    conv15 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv15')(conv14)	
    conv16 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv16')(conv15)	
    conv17 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv17')(conv16)	
    conv18 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv18')(conv17)	
    conv19 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv19')(conv18)	
    conv20 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv20')(conv19)	
    conv21 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv21')(conv20)	
    conv22 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv22')(conv21)	
    conv23 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv23')(conv22)	
    conv24 = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv24')(conv23)
    output = Conv1D(1,20,padding='same',strides=1,activation='linear',name='conv25')(conv24)
    return Model(input_layer,output)
	

conv_model1 = conv_model()

conv_model1.summary()



lr = 1e-3
reduce_lr = ReduceLROnPlateau(factor=0.8,monitor='loss',mode='auto',patience=10,min_lr=1e-9)
conv_checkpoint = ModelCheckpoint(model_save_path,monitor='loss',mode='auto',save_best_only='True')

conv_model1.compile(loss='mse',optimizer=Adam(lr=lr))
conv_model1.fit(X,y,epochs=2000,batch_size=32,verbose=1,
        callbacks=[reduce_lr,EarlyStopping(monitor='loss',patience=30,mode='auto'),conv_checkpoint])


'''
conv_model2 = conv_model(block_number=500,filter_size=10)
conv_model2.summary()
reduce_lr = ReduceLROnPlateau(factor=0.8,monitor='loss',mode='auto',patience=10,min_lr=1e-9)
conv_checkpoint2 = ModelCheckpoint(model_save_path2,monitor='loss',mode='auto',save_best_only='True')

conv_model2.compile(loss='mse',optimizer=Adam(lr=lr))
conv_model2.fit(X,y,epochs=2000,batch_size=32,verbose=1,
        callbacks=[reduce_lr,EarlyStopping(monitor='loss',patience=30,mode='auto'),conv_checkpoint2])

'''
