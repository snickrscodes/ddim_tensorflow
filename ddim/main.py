from model import DiffusionModel
import tensorflow as tf
import math
import numpy as np
import datetime
import tensorflow_datasets as tfds

params = dict(
    name='ddim', # the model name
    chkpt_dir='C:/Users/saraa/Desktop/ddpm/checkpoints/', # where to save checkpoints
    image_dir='C:/Users/saraa/Desktop/ddpm/images/', # where to save checkpoints
    block_depth=3, # how many residual blocks per down/up block
    attn_res=[[8, 8]], # resolutions to apply attention to
    dropout=0.1, # global dropout rate
    use_conv_samp=False, # toggles conv resampling in up/down blocks
    input_res=[32, 32, 1], # input resolution (HEIGHT BY WIDTH)
    ch_fac=[1, 1, 1, 2, 2, 4, 4, 8, 8, 16, 16], # the factor to scale channels by
    model_depth=3, # how many times to down/upsample
    emb_dim=32, # embedding dimension
    lr=0.001, # model learning rate
    T=20, # number of diffusion steps
    sigma_t=0.0, # the noise rate (when it = 0, the model is deterministic)
    batch_size=32, # batch size for training
    beta_1=0.9, # adam optimization parameter
    beta_2=0.999, # adam optimization parameter
    ema_weight=0.995, # exponential moving average weight
    min_t=2, # min time to sample from
    max_t=18, # max time to sample from
)

(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

def normalize_img(image, label):
  return tf.image.resize(tf.cast(image, tf.float32) / 255.0, [32, 32])

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(32)
model = DiffusionModel(params)
for i in range(10):
    for entry in ds_train:
        model.train(entry)
    print(f'{datetime.datetime.now()} completed {model.steps} training steps')
    model.save()
    model.generate_samples(16)
model.save()
