import tensorflow as tf
import numpy as np
import math

class Layer(object):
    def __init__(self, name: str, activation=None, input_shape=None):
        self.name = name
        self.activation = activation if activation is not None else lambda x: x
        self.input_shape = input_shape
        self.variables = []
        self.training = True
        if self.input_shape is not None:
            self.init_vars()
    def __call__(self, input: tf.Tensor):
        return input
    def init_vars(self):
        pass
    def set_training(self, training):
        self.training = training
    def format_padding(self, padding, ndim=2):
        ps = [[0, 0] for _ in range(ndim)]
        if isinstance(padding, (tuple, list)):
            for i in range(len(padding)):
                if isinstance(padding[i], (tuple, list)):
                    ps[i] = padding[i]
                else:
                    ps[i] = [padding[i], padding[i]]
        elif isinstance(padding, int):
            ps = [[padding, padding] for _ in range(ndim)]
        if not isinstance(padding, str):
            return [[0, 0], *ps, [0, 0]]
        return padding # in this case string padding
    def format_arg(self, arg, ndim=2):
        if isinstance(arg, int):
            return [arg]*ndim
        return arg

class Linear(Layer):
    def __init__(self, units: int, name: str, activation=None, input_shape=None):
        super().__init__(name, activation, input_shape)
        self.units = units
        self.weight, self.bias = None, None

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.input_shape = input.shape.as_list()
            self.init_vars()
        return self.activation(tf.matmul(input, self.weight) + self.bias)

    def init_vars(self, variance=2.0):
        fan_in = np.prod(self.input_shape[1:])
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[self.input_shape[-1], self.units], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=[1, self.units]), name=self.name+"_bias", trainable=True)
        self.variables = [self.weight, self.bias]

    def load_variables(self, vars):
        for name, var in vars.items():
            if self.name+'_weight' in name:
                self.weight = var
            elif self.name+'_bias' in name:
                self.bias = var
        self.variables = [self.weight, self.bias]


class GroupNormalization(Layer):
    def __init__(self, name: str, groups=32, axis=-1, epsilon=1.0e-6, activation=None, input_shape=None):
        super().__init__(name, activation, input_shape)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.gamma, self.beta = None, None

    def __call__(self, input: tf.Tensor):
        if self.gamma is None or self.beta is None:
            self.input_shape = input.shape.as_list()
            self.init_vars()
        # input should be in nhwc
        n, h, w, c = input.shape.as_list()
        if c % self.groups != 0:
            raise ValueError(f"channels {c} not divisible by groups {self.groups}")
        reshaped_inputs = tf.reshape(input, [n, h, w, self.groups, c // self.groups]) # [n, h, w, g, c_g]
        mean, variance = tf.nn.moments(reshaped_inputs, axes=[1, 2, 4], keepdims=True)
        x_norm = tf.reshape((reshaped_inputs - mean) / tf.sqrt(variance + self.epsilon), [n, h, w, c])
        return self.gamma * x_norm + self.beta

    def init_vars(self):
        # channel dimension
        param_shape = [1] * len(self.input_shape)
        param_shape[self.axis] = self.input_shape[self.axis]
        self.gamma = tf.Variable(tf.ones(shape=param_shape), name=self.name+"_gamma", trainable=True)
        self.beta = tf.Variable(tf.zeros(shape=param_shape), name=self.name+"_beta", trainable=True)
        self.variables = [self.gamma, self.beta]

    def load_variables(self, vars):
        for name, var in vars.items():
            if self.name+'_gamma' in name:
                self.gamma = var
            elif self.name+'_beta' in name:
                self.beta = var
        self.variables = [self.gamma, self.beta]

class Conv2D(Layer):
    def __init__(self, filters: int, name: str, kernel=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation=None, input_shape=None):
        super().__init__(name, activation, input_shape)
        self.filters = filters
        self.kernel = self.format_arg(kernel)
        self.stride = self.format_arg(stride)
        self.dilation = self.format_arg(dilation)
        self.padding = self.format_padding(padding)
        self.weight, self.bias = None, None

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.input_shape = input.shape.as_list()
            self.init_vars()
        return self.activation(tf.nn.conv2d(input, self.weight, self.stride, self.padding, "NHWC", self.dilation) + self.bias)

    # using HWIO weight format and NHWC
    def init_vars(self, variance=2.0):
        fan_in = np.prod(self.input_shape[1:]) # numpy automatically converts
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel, self.input_shape[-1], self.filters], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=[1, 1, 1, self.filters]), name=self.name+"_bias", trainable=True)
        self.variables = [self.weight, self.bias]
    
    def load_variables(self, vars):
        for name, var in vars.items():
            if self.name+'_weight' in name:
                self.weight = var
            elif self.name+'_bias' in name:
                self.bias = var
        self.variables = [self.weight, self.bias]
    
    # for sanity checking
    def calculate_output_shape(self):
        oh, ow = 0, 0
        if self.padding == 'SAME':
            oh = math.ceil(float(self.input_shape[1])/float(self.stride[0]))
            ow = math.ceil(float(self.input_shape[2])/float(self.stride[1]))
        elif self.padding == 'VALID':
            oh = math.ceil(float(self.input_shape[1]-self.kernel[0]+1)/float(self.stride[0]))
            ow  = math.ceil(float(self.input_shape[2]-self.kernel[1]+1)/float(self.stride[1]))
        else:
            oh = int((self.input_shape[1]+self.padding[1][0]+self.padding[1][1]-self.dilation[0]*(self.kernel[0]-1)-1)/self.stride[0]+1)
            ow = int((self.input_shape[2]+self.padding[2][0]+self.padding[2][1]-self.dilation[1]*(self.kernel[1]-1)-1)/self.stride[1]+1)
        return [-1, oh, ow, self.filters]

class Conv2DTranspose(Layer):
    def __init__(self, filters: int, name: str, kernel=(2, 2), output_shape=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation=None, input_shape=None):
        super().__init__(name, activation, input_shape)
        self.filters = filters
        self.kernel = self.format_arg(kernel)
        self.stride = self.format_arg(stride)
        self.dilation = self.format_arg(dilation)
        self.padding = self.format_padding(padding)
        self.output_shape = output_shape
        self.weight, self.bias = None, None

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.input_shape = input.shape.as_list()
            self.init_vars()
        # need to redefine it just in case batch size changes
        self.output_shape[0] = input.shape.as_list()[0]
        return self.activation(tf.nn.conv2d_transpose(input, self.weight, self.output_shape, self.stride, self.padding, "NHWC", self.dilation) + self.bias)

    def init_vars(self, variance=2.0):
        fan_in = np.prod(self.input_shape[1:])
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[*self.kernel, self.filters, self.input_shape[-1]], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=[1, 1, 1, self.filters]), name=self.name+"_bias", trainable=True)
        self.variables = [self.weight, self.bias]
        # need output shape
        oh, ow = 1, 1
        if self.padding == 'SAME':
            oh = self.input_shape[1]*self.stride[0]
            ow = self.input_shape[2]*self.stride[1]
        elif self.padding == 'VALID':
            oh = (self.input_shape[1]-1)*self.stride[0]+self.dilation[0]*(self.kernel[0]-1)+1
            ow = (self.input_shape[2]-1)*self.stride[1]+self.dilation[1]*(self.kernel[1]-1)+1
        else:
            oh = self.stride[0]*(self.input_shape[1]-1)+self.kernel[0]+(self.kernel[0]-1)*(self.dilation[0]-1)-self.padding[1][0]-self.padding[1][1]
            ow = self.stride[1]*(self.input_shape[2]-1)+self.kernel[1]+(self.kernel[1]-1)*(self.dilation[1]-1)-self.padding[2][0]-self.padding[2][1]
        if self.output_shape is None:
            self.output_shape = [self.input_shape[0], oh, ow, self.filters]

    def load_variables(self, vars):
        for name, var in vars.items():
            if self.name+'_weight' in name:
                self.weight = var
            elif self.name+'_bias' in name:
                self.bias = var
        self.variables = [self.weight, self.bias]

class AttentionLayer(Layer):
    def __init__(self, name: str, d_model=8, dropout=0.1):
        super().__init__(name) # this impl is diff, we don't split inputs between any heads
        self.d_model = d_model # number of input channels
        self.dropout = dropout
        self.norm = GroupNormalization(self.name+'_norm')
        self.q = Conv2D(d_model, self.name+'_q_lin', 1) # ks = 1 in all instances for channel projection
        self.k = Conv2D(d_model, self.name+'_k_lin', 1)
        self.v = Conv2D(d_model, self.name+'_v_lin', 1)
        self.out_proj = Conv2D(d_model, self.name+'_out_proj', 1)
        self.layers = [self.norm, self.q, self.k, self.v, self.out_proj]
    
    def load_variables(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)
            self.variables.extend(layer.variables)

    def set_training(self, training):
        for layer in self.layers:
            layer.set_training(training)

    def __call__(self, input: tf.Tensor):
        # basically just scaled dot product attention but with a residual connection
        x = self.norm(input)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        n, h, w, c = q.shape.as_list()
        q = tf.reshape(q, [n, h*w, c])  # [n, hw, c]
        k = tf.reshape(k, [n, h*w, c])
        v = tf.reshape(v, [n, h*w, c])
        scores = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) # [n, hw, c] * [n, c, hw] = [n, hw, hw]
        weights = tf.nn.softmax(scores/math.sqrt(c), axis=-1) # dk = c in this case
        if self.training:
            weights = tf.nn.dropout(weights, rate=self.dropout)
            # mask = tf.cast(tfp.distributions.Bernoulli(probs=1.0-self.dropout).sample(sample_shape=weights.shape.as_list()), dtype=tf.float32)
            # weights *= mask/(1.0-self.dropout)
        x = tf.matmul(weights, v) # [n, hw, hw] * [n, hw, c] = [n, hw, c]
        x = tf.reshape(x, [n, h, w, c])
        x = self.out_proj(x)
        if len(self.variables) == 0:
            for layer in self.layers:
                self.variables.extend(layer.variables)
        return input+x
    
class Upsample(Layer):
    def __init__(self, name: str, d_model: int, use_conv=True):
        super().__init__(name)
        self.d_model = d_model # number of input channels
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = Conv2D(d_model, self.name+'_conv', kernel=3, padding='SAME')
        
    def load_variables(self, vars: dict):
        if self.use_conv:
            self.conv.load_variables(vars)
            self.variables.extend(self.conv.variables)

    def set_training(self, training):
        if self.use_conv:
            self.conv.set_training(training)

    def __call__(self, input: tf.Tensor):
        rank = len(input.shape.as_list())
        # upscale spatial dimensions by factor of 2
        x = input
        for i in range(1, rank-1):
            x = tf.repeat(x, repeats=2, axis=i) # NHWC, NDHWC etc
        if self.use_conv:
            x = self.conv(x)
            if len(self.variables) == 0:
                self.variables.extend(self.conv.variables)
        return x
    
class Downsample(Layer):
    def __init__(self, name: str, d_model: int, use_conv=True):
        super().__init__(name)
        self.d_model = d_model # number of input channels
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = Conv2D(d_model, self.name+'_conv', kernel=3, stride=2, padding=[[0, 1], [0, 1]])
        
    def load_variables(self, vars: dict):
        if self.use_conv:
            self.conv.load_variables(vars)
            self.variables.extend(self.conv.variables)

    def set_training(self, training):
        if self.use_conv:
            self.conv.set_training(training)

    def __call__(self, input: tf.Tensor):
        if self.use_conv:
            x = self.conv(input)
            if len(self.variables) == 0:
                self.variables.extend(self.conv.variables)
            return x
        else:
            return tf.nn.avg_pool(input=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

class Resnet(Layer):
    def __init__(self, name: str, in_channels: int, out_channels=0, dropout=0.1):
        super().__init__(name)
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == 0 else out_channels
        self.norm1 = GroupNormalization(self.name+'_norm1')
        self.conv1 = Conv2D(self.out_channels, self.name+'_conv1', 3, 1, 1, 1)
        self.emb_lin = Linear(self.out_channels, self.name+'_emb_lin')
        self.norm2 = GroupNormalization(self.name+'_norm2')
        self.conv2 = Conv2D(self.out_channels, self.name+'_conv2', 3, 1, 1, 1)
        self.layers = [self.norm1, self.conv1, self.emb_lin, self.norm2, self.conv2]
        self.nin_shortcut = None
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2D(self.out_channels, self.name+'_nin_shortcut', 1, 1, 0, 1) # pointwise convolution
            self.layers.append(self.nin_shortcut)

    def load_variables(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)
            self.variables.extend(layer.variables)

    def set_training(self, training):
        for layer in self.layers:
            layer.set_training(training)

    def __call__(self, input: tf.Tensor, temb: tf.Tensor):
        h = self.norm1(input)
        h = tf.nn.swish(h)
        h = self.conv1(h)
        h = h + tf.transpose(self.emb_lin(tf.nn.swish(temb))[:, :, tf.newaxis, tf.newaxis], perm=[0, 2, 3, 1]) # must convert the embedding to NHWC
        h = self.norm2(h)
        h = tf.nn.swish(h)
        if self.training:
            h = tf.nn.dropout(h, self.dropout)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            input = self.nin_shortcut(input)
        if len(self.variables) == 0:
            for layer in self.layers:
                self.variables.extend(layer.variables)
        return input+h