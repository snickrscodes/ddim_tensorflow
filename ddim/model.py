from layers import *
from optimization import Adam
import tensorflow as tf
import math
from PIL import Image
import os

class DownBlock(Layer):
    def __init__(self, name: str, in_channels: int, out_channels: int, n_res_layers: int, use_attn=False, dropout=0.1, use_conv=False, down_samp=True):
        super().__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_res_layers = n_res_layers
        self.use_conv = use_conv # for the downsampling layer
        self.down_samp = down_samp # should always be true unless the last downsampling block
        self.downsample = None
        if self.down_samp:
            self.downsample = Downsample(self.name+'_downsample', self.out_channels, self.use_conv)
        self.dropout = dropout
        self.use_attn = use_attn # if to use attention in this block
        self.res_layers, self.attn_layers = [], []
        for i in range(self.n_res_layers):
            self.res_layers.append(Resnet(self.name+'_resblock'+str(i), self.in_channels if i == 0 else self.out_channels, self.out_channels, self.dropout))
            if self.use_attn:
                self.attn_layers.append(AttentionLayer(self.name+'attn'+str(i), self.out_channels, self.dropout))
        
    def load_variables(self, vars: dict):
        for layer in self.res_layers:
            self.variables.extend(layer.variables)
        for layer in self.attn_layers:
            layer.load_variables(vars)
            self.variables.extend(layer.variables)
        if self.downsample is not None:
            self.downsample.load_variables(vars)
            self.variables.extend(self.downsample.variables)

    def set_training(self, training):
        for layer in self.res_layers:
            layer.set_training(training)
        for layer in self.attn_layers:
            layer.set_training(training)
        if self.downsample is not None:
            self.downsample.set_training(training)

    def __call__(self, input: tf.Tensor, temb: tf.Tensor):
        x = input
        states = []
        for i in range(self.n_res_layers):
            x = self.res_layers[i](x, temb)
            if self.use_attn:
                x = self.attn_layers[i](x)
            states.append(x)
        if self.down_samp:
            x = self.downsample(x)
            states.append(x)
        if len(self.variables) == 0:
            for layer in self.res_layers:
                self.variables.extend(layer.variables)
            for layer in self.attn_layers:
                self.variables.extend(layer.variables)
            if self.downsample is not None:
                self.variables.extend(self.downsample.variables)
        return x, states

class UpBlock(Layer):
    def __init__(self, name: str, in_channels: int, skip_in: int, out_channels: int, n_res_layers: int, use_attn=False, dropout=0.1, use_conv=False, up_samp=True):
        super().__init__(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_in = skip_in # the number of output channels from the down block
        self.n_res_layers = n_res_layers
        self.up_samp = up_samp # should always be true unless the first upsampling block
        self.use_conv = use_conv
        self.upsample = None
        if self.up_samp:
            self.upsample = Upsample(self.name + '_upsample', self.out_channels, self.use_conv)
            self.variables.extend(self.upsample.variables)
        self.dropout = dropout
        self.use_attn = use_attn
        self.res_layers, self.attn_layers = [], []
        for i in range(self.n_res_layers+1):
            self.res_layers.append(Resnet(self.name + '_resblock' + str(i), self.in_channels+self.skip_in if i == 0 else self.out_channels+self.skip_in, self.out_channels, self.dropout))
            if self.use_attn:
                self.attn_layers.append(AttentionLayer(self.name + '_attn' + str(i), self.out_channels, self.dropout))
    
    def load_variables(self, vars: dict):
        for layer in self.res_layers:
            self.variables.extend(layer.variables)
        for layer in self.attn_layers:
            layer.load_variables(vars)
            self.variables.extend(layer.variables)
        if self.upsample is not None:
            self.upsample.load_variables(vars)
            self.variables.extend(self.upsample.variables)

    def set_training(self, training):
        for layer in self.res_layers:
            layer.set_training(training)
        for layer in self.attn_layers:
            layer.set_training(training)
        if self.upsample is not None:
            self.upsample.set_training(training)
    
    def __call__(self, input: tf.Tensor, temb: tf.Tensor, states: list[tf.Tensor]):
        x = input
        for i in range(self.n_res_layers+1):
            x = self.res_layers[i](tf.concat([x, states.pop()], axis=-1), temb)
            if self.use_attn:
                x = self.attn_layers[i](x)
        if self.up_samp:
            x = self.upsample(x)
        if len(self.variables) == 0:
            for layer in self.res_layers:
                self.variables.extend(layer.variables)
            for layer in self.attn_layers:
                self.variables.extend(layer.variables)
            if self.upsample is not None:
                self.variables.extend(self.upsample.variables)
        return x
    
class TimestepEmbedding(Layer):
    def __init__(self, name: str, dim: int):
        super().__init__(name)
        self.dim = dim # the embedding dimension (will get scaled by dense layers)
        self.emb_ch = self.dim*4 # the final embedding dimension
        self.d0 = Linear(self.emb_ch, self.name+'_dense0', tf.nn.swish)
        self.d1 = Linear(self.emb_ch, self.name+'_dense1')

    def load_variables(self, vars: dict):
        self.d0.load_variables(vars)
        self.d1.load_variables(vars)
        self.variables.extend(self.d0.variables)
        self.variables.extend(self.d1.variables)
    
    def __call__(self, timesteps: tf.Tensor):
        # matches the implementation of timestep embeddings in DDPM/DDIM
        half_dim = self.dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.cast(timesteps[:, None], tf.float32) * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
        if self.dim % 2 == 1:
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        emb = self.d0(emb)
        emb = self.d1(emb)
        if len(self.variables) == 0:
            self.variables.extend(self.d0.variables)
            self.variables.extend(self.d1.variables)
        return emb


class Model(object): # denoising diffusion implicit model implementation
    def __init__(self, params: dict):
        # import param dictionary
        self.name = params['name']
        self.CHKPT_DIR = params['chkpt_dir']
        self.block_depth = params['block_depth']
        self.attn_res = params['attn_res']
        self.dropout = params['dropout']
        self.use_conv_samp = params['use_conv_samp']
        self.input_res = params['input_res']
        self.model_depth = params['model_depth']
        self.ch_mult = params['ch_fac'][:self.model_depth+1]
        self.emb_dim = params['emb_dim']
        self.beta_1 = params['beta_1']
        self.beta_2 = params['beta_2']
        self.ema_weight = params['ema_weight']
        self.lr = params['lr']
        self.optimizer = Adam(self.lr, self.beta_1, self.beta_2)
        # build the model
        self.variables = []
        self.ema = {}
        self.layers = []
        self.build()

    def load_variables(self, vars):
        for layer in self.layers:
            layer.load_variables(vars)
            self.variables.extend(layer.variables)

    def build(self):
        self.timestep_embedding = TimestepEmbedding(self.name+'_timestep_embedding', self.emb_dim)
        self.conv_in = Conv2D(self.emb_dim, self.name+'_conv_in', 3, 1, 1, 1)
        self.layers.append(self.timestep_embedding) # first timestep embedding
        self.layers.append(self.conv_in) # conv transition to down blocks
        res = [self.input_res[0], self.input_res[1]]
        fin_ch = 0
        # downsampling layers
        self.down_blocks = []
        for i in range(self.model_depth):
            block = DownBlock(self.name+'_down'+str(i), self.emb_dim*self.ch_mult[i], self.emb_dim*self.ch_mult[i+1], self.block_depth, res in self.attn_res, self.dropout, self.use_conv_samp, i < self.model_depth-1)
            self.layers.append(block)
            self.down_blocks.append(block)
            fin_ch = self.emb_dim*self.ch_mult[i+1]
            res[0] //= 2
            res[1] //= 2
        # middle layers
        self.mid_res0 = Resnet(self.name+'_mid_res0', fin_ch, fin_ch, self.dropout)
        self.mid_attn = AttentionLayer(self.name+'_mid_attn', fin_ch, self.dropout)
        self.mid_res1 = Resnet(self.name+'_mid_res1', fin_ch, fin_ch, self.dropout)
        self.layers.append(self.mid_res0)   
        self.layers.append(self.mid_attn)
        self.layers.append(self.mid_res1)
        # upsampling layers
        self.up_blocks = []
        for i in range(self.model_depth):
            block = UpBlock(self.name+'_up'+str(i), self.emb_dim*self.ch_mult[-(i+1)], self.emb_dim*self.ch_mult[-(i+1)], self.emb_dim*self.ch_mult[-(i+2)], self.block_depth, res in self.attn_res, self.dropout, self.use_conv_samp, i < self.model_depth-1)
            self.layers.append(block)
            self.up_blocks.append(block)
            res[0] *= 2
            res[1] *= 2
        # final layers
        self.end_norm = GroupNormalization(self.name+'_end_norm')
        self.end_conv = Conv2D(self.input_res[2], self.name+'_out_conv', 3, 1, 1, 1)
        self.layers.append(self.end_norm)
        self.layers.append(self.end_conv)
        # data norm variables
        moment_shape = [1] * (len(self.input_res)+1)
        moment_shape[-1] = self.input_res[2]
        self.ema[self.name+'_data_moving_mean'] = self.data_moving_mean = tf.Variable(tf.zeros(moment_shape), trainable=False, name=self.name+'_data_moving_mean')
        self.ema[self.name+'_data_moving_var'] = self.data_moving_var = tf.Variable(tf.ones(moment_shape), trainable=False, name=self.name+'_data_moving_var')

    def update_ema(self):
        # update ema
        for var in self.variables:
            if var.name not in self.ema and var.dtype is tf.float32:
                self.ema[var.name] = tf.Variable(tf.zeros_like(var), trainable=False, name=var.name)
            self.ema[var.name].assign(self.ema[var.name] * self.ema_weight + (1 - self.ema_weight) * var)

    def __call__(self, input, t):
        temb = self.timestep_embedding(t)
        x = x_i = self.conv_in(input)
        states = [x_i]
        for block in self.down_blocks:
            x, state = block(x, temb)
            states += state
        states = [states[i:i+self.block_depth+1] for i in range(0, len(states), self.block_depth+1)]
        x = self.mid_res0(x, temb)
        x = self.mid_attn(x)
        x = self.mid_res1(x, temb)
        for block in self.up_blocks:
            x = block(x, temb, states.pop())
        x = self.end_norm(x)
        x = self.end_conv(x)
        if len(self.variables) == 0:
            for layer in self.layers:
                self.variables.extend(layer.variables)
        return x
    
    def set_training(self, training):
        for layer in self.layers:
            layer.set_training(training)
        if training is False:
            self.load_variables(self.ema)
        else:
            vars = {}
            for var in self.variables:
                vars[var.name] = var
            self.load_variables(self.vars)

    def save_checkpoint(self):
        if not os.path.exists(self.CHKPT_DIR):
            os.makedirs(self.CHKPT_DIR)
        vars = {}
        for var in self.variables:
            vars[var.name] = var
        checkpoint = tf.train.Checkpoint(**vars)
        size = str(len(os.listdir(self.CHKPT_DIR)))
        newpath = self.CHKPT_DIR+'chkpt'+size+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        checkpoint.save(newpath+'checkpt'+size)
        self.save_ema(self.CHKPT_DIR)

    def load_checkpoint(self, index=-1):
        vars = {}
        if index < 0:
            index = str(len(os.listdir(self.CHKPT_DIR))+index)
        else:
            index = str(index)
        checkpoint_reader = tf.train.load_checkpoint(self.CHKPT_DIR+'chkpt'+index+'/')
        var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map:
            name = var_name[:str.find(var_name, '/')]
            variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
            if variable.dtype is tf.float32:
                vars[name] = variable
        self.load_variables(vars)
        self.load_ema(self.CHKPT_DIR)

    def save_ema(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        checkpoint = tf.train.Checkpoint(**self.ema)
        size = str(len(os.listdir(dir))-1)
        newpath = dir+'chkpt'+size+'/ema_chkpt/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        checkpoint.save(newpath+'ema_checkpt'+size)

    def load_ema(self, dir, index=-1):
        vars = {}
        if index < 0:
            index = str(len(os.listdir(dir))+index)
        else:
            index = str(index)
        checkpoint_reader = tf.train.load_checkpoint(dir+'chkpt'+index+'/ema_chkpt/')
        var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map:
            name = var_name[:str.find(var_name, '/')]
            variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
            if variable.dtype is tf.float32:
                vars[name] = variable
        self.ema = vars

class DiffusionModel(object): # a wrapper for the DDIM to control inference and training
    def __init__(self, params: dict):
        self.model = Model(params)
        self.T = params['T']
        self.sigma_t = params['sigma_t']
        self.input_res = params['input_res']
        self.image_dir = params['image_dir']
        self.min_t = params['min_t']
        self.max_t = params['max_t']
        self.ema_weight = params['ema_weight']
        self.training = True
        self.steps = 0
        self.model.build()
    
    def save(self):
        self.model.save_checkpoint()
    
    def load(self):
        self.model.load_checkpoint()

    def set_training(self, training):
        self.training = training
        self.model.set_training(training)

    def signal_rate(self, times, s=1.0e-5):
        # returns the cumulative product of alpha at time step t
        def f(t):
            if isinstance(t, tf.Tensor):
                t = tf.clip_by_value(t, 0, self.T)
                f_t = tf.square(tf.cos((t / self.T + s) / (1.0 + s) * math.pi / 2.0))
                f_t = f_t[:, tf.newaxis, tf.newaxis, tf.newaxis]
                return tf.cast(f_t, tf.float32)
            else: # scalar values
                f_t = math.cos((t / self.T + s) / (1.0 + s) * math.pi / 2.0) ** 2.0
                return f_t
        return f(times) / f(0.0) # assuming we start when t = 0 of course
    
    def forward_diffusion(self, x0: tf.Tensor, t: tf.Tensor): # times is a 1d tensor of diffusion times
        a_t_prod = self.signal_rate(t)
        eps = tf.random.normal(x0.shape, mean=0.0, stddev=1.0)
        return tf.sqrt(a_t_prod)*x0 + tf.sqrt(1.0-a_t_prod)*eps, eps
    
    def denoise(self, x_t: tf.Tensor, t: tf.Tensor): # t is a list of diffusion times
        a_t_prod = self.signal_rate(t)
        a_tp_prod = self.signal_rate(t-1) # previous signal rate
        pred_noise = self.model(x_t, t)
        px_0 = (x_t - tf.sqrt(1.0-a_t_prod)*pred_noise)/tf.sqrt(a_t_prod) # predicted x_0
        dx_t = tf.sqrt(1-a_tp_prod-self.sigma_t**2.0)*pred_noise # direction to x_t
        x_p = tf.sqrt(a_tp_prod)*px_0 + dx_t
        if self.sigma_t > 0: # when sigma_t = 0, it's deterministic
            x_p += tf.random.normal(x_t.shape, mean=0, stddev=self.sigma_t) # random noise
        return x_p

    def train(self, images: tf.Tensor):
        b, h, w, c = images.shape.as_list()
        with tf.GradientTape() as tape:
            t = tf.random.uniform([b], self.min_t, self.max_t, dtype=tf.int32)
            x = self.normalize(images, True)
            x_t, true_noise = self.forward_diffusion(x, t)
            pred_noise = self.model(x_t, t)
            loss = tf.reduce_mean(tf.square(pred_noise - true_noise))
        gradients = tape.gradient(loss, self.model.variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.variables))
        self.model.update_ema()
        self.steps += 1

    def normalize(self, images, upd_ema=False): # model a gaussian - noise dist of choice
        mean, var = tf.nn.moments(images, axes=[0, 1, 2]) # reduce all but channel dimension
        if self.training and upd_ema:
            self.model.data_moving_mean.assign(self.model.data_moving_mean*self.ema_weight+mean*(1.0-self.ema_weight))
            self.model.data_moving_var.assign(self.model.data_moving_var*self.ema_weight+var*(1.0-self.ema_weight))
        return (images - mean) / tf.sqrt(var+1.0e-6)

    def denormalize(self, images, eps=1.0e-3):
        x = images * tf.sqrt(self.model.data_moving_var+1.0e-6) + self.model.data_moving_mean
        return tf.clip_by_value(x, 0.0+eps, 1.0-eps) # 0-1 for images
    
    # unused
    def normalize_images(self, image_data):
        min_vals = tf.reduce_min(image_data, axis=[1, 2, 3], keepdims=True)
        max_vals = tf.reduce_max(image_data, axis=[1, 2, 3], keepdims=True)
        return (image_data - min_vals) / (max_vals - min_vals) # also 0-1

    def generate_samples(self, n_samples: int, eps=1.0e-3, filename='image'):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        x_t = tf.random.normal([n_samples, *self.input_res], mean=0.0, stddev=1.0)
        # reverse diffusion
        for t in range(self.T, 0, -1):
            times = tf.fill([n_samples], t)
            x_t = self.denoise(x_t, times)
        x_t = self.denormalize(x_t)
        x_t = tf.clip_by_value(x_t, 0+eps, 1-eps) * 255
        images = x_t.numpy().astype(np.uint8)
        count = len(os.listdir(self.image_dir))
        for image in images:
            image = Image.fromarray(np.squeeze(image, axis=-1), mode='L')
            image.save(self.image_dir+filename+str(count)+'.jpg')
            count += 1