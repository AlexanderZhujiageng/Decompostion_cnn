from keras import backend as K
from keras import activations,initializers,regulizers,constraints
from keras.layers import Lambda,Layer,InputSpec,Convolution1D,Convolution2D,add,multiply,Activation,Input,concatenate
from keras.layers.convolutional import _Conv
from keras.layers.merge import _Merge
from keras.utils import conv_utils
from keras.model import Model
import numpy as np
from init import ComplexInit,ComplexIndependentFilters



def sanitizedInitGet(init):
    if init in ['complex','complex_independent','glorot_complex','he_complex']:
        return init
    else:
        return initializers.get(init)

def santizedInitSer(init):
    if init == 'complex' or isinstance(init,ComplexInit):
        return 'complex'
    elif init =='complex_independent' of isinstance(init,ComplexIndependentFilters):
        return 'complex_independet'
    else:
        return initializers.serialize(init)

class ComplexConv2D(Layer):
    def __init__(self,rank=2,
            filters,
            kernel_size,
            strides=1,
            padding='same',
            data_format=None,
            dilation_rate = 1
            activation=None,
            use_bias=False,
            kernel_initializer='normal',
            kernel_regulizer=None,
            bias_regulizer=None,
            kernel_constriant=None,
            bias_constriant=None,
            seed=None,
            epsilon=1e-7,
            **kwargs):
        super(ComplexConv2D,self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size,rank,'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides,rank,'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format == conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate,rank,'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = sanitizedInitGet(kernel_initializer)
        self.kernel_regulizer = regulizers.get(kernel_regulizer)
        self.kernel_constriant = constriants.get(kernel_constriant)
        if seed is None:
            self.seed - np.random.randint(1,10e6)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self,input_shape):
        if self.data_format = 'channel_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('the channel dimension should be defined. Found None')
        input_dim = input_shape[channel_axis] //2
        self.kernel_shape = self.kernel_size + (input_dim,self.filters)
        self.kernel = self.add_weight(
                self.kernel_shape,
                initializer = self.kernel_initializer,
                name='kernel',
                regulizer = self.kernel_regulizer,
                constraint = self.kernel_constriant)
        self.input_spec = InputSpec(ndim=self.rank + 2,
                axes={channel_axis:input_dim * 2})
        self.built = True

    def call(self,inputs):
        if self.rank == 2:
            f_real = self.kernel[:,:,:self.filters]
            f_imag = self.kernel[:,:,self.filters:]

        convArgs = {'strides':       self.strides,
                    'padding':       self.padding,
                    'data_format':   self.data_format,
                    'dilation_rate': self.dilation_rate
                    }
        convFunc = {1:K.conv1d,
                    2:K.conv2d,
                    3:K.conv3d}[self.rank]

        f_real._keras_shape = self.kernel_shape
        f_imag._keras_shape = self.kernel_shape

        cat_kernels_4_real = K.concatenate([f_real,f_imag],axis=-2)
        cat_kernels_4_imag = K.concatenate([f_imag,f_real],axis=-2)
        cat_kernels_4_complex = K.concatenate([cat_kernels_4_real,cat_kernels_4_imag],axis=-1)
        cat_kernels_4_complex._keras_shape = self.kernel_size + (2* input_dim, 2* self.filters)
        output = convFunc(inputs,cat_kernels_4_complex,**convArgs)

        if self.activation is not None:
            output = self.activation(output)
        return output


    def compute_output_shape(self,input_shape):
        if self.data_format =='channel_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding = self.padding,
                        stride = self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (2*self.filters,)
        if self.data_format == 'channel_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + (2*self.filters,) + tuple(new_space)
