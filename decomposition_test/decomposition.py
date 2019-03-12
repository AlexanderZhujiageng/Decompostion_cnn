import numpy as np
import keras
from keras.layers import Input,Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
import pdb

model_save_path = './model/improvement-{epochs:02d}-{loss:.5f}.hdf5'
with_strides_model_path ='.strides_model/improvement-{epochs:02d}-{loss:.5f}.hdf5'

width = 100
height = 100
'''
inputs_path = 'inputs.txt'
outputs_path ='outputs.txt'

X = np.loadtxt(inputs_path,delimiter=',').reshape(len(X),width,height,1)
y = np.loadtxt(outpus_path,delimiter=',').reshape(len(y),width,height,1)
'''

def conv_model(block_number=20,filter_size=5,strides=1):
    input_layer = Input(shape=(100,100,1))
    for i in range(block_number):
        if i ==0:
            x = Conv2D(1,filter_size,padding='same',strides=strides,activation='linear',name='conv'+str(i+1))(input_layer)
        elif i!=block_number-1:
            x = Conv2D(1,filter_size,padding='same',strides=strides,activation='linear',name='conv'+str(i+1))(x)
        else:
            output = Conv2D(1,filter_size,padding='same',strides=strides,activation='linear',name='output')(x)

    model = Model(input_layer,output)
    return model

conv_model1 = conv_model(block_number=24)
conv_strides_model = conv_model(block_number=12,strides=2)

conv_model1.summary()

pdb.set_trace()

conv_strides_model.summary()



    

