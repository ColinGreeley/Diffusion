import math
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import collections
import os
import json
import pandas as pd
import gc
from tqdm import tqdm
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
#import tensorflow_probability as tfp
import keras_nlp
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from tokenizers import BertWordPieceTokenizer
#from keras import layers


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

num_epochs = 10000 
image_size = 256
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 30

# sampling
min_signal_rate = 0.02   # 0.02
max_signal_rate = 0.98   # 0.95
s1 = 5

# architecture
embedding_max_frequency = 1000.0
widths = [128, 256, 512, 1024] # [128, 64, 32, 16, 8]
d_model = 768
block_depths = [2, 3, 4, 5]
heads = 4

# optimization
batch_size = 9
ema = 0.999
learning_rate = 1e-6
weight_decay = 1e-6




def get_data():
    trainval_image_dir = "./Data/train2014/train2014/"
    trainval_captions_dir = "./Data/annotations_trainval2014/annotations"
    test_image_dir = "./Data/val2017/val2017/"
    test_captions_dir = "./Data/annotations_trainval2017/annotations"
    trainval_captions_filepath = os.path.join(trainval_captions_dir, 'captions_train2014.json')
    test_captions_filepath = os.path.join(test_captions_dir, 'captions_val2017.json')
    all_filepaths = np.array([os.path.join(trainval_image_dir, f) for f in os.listdir(trainval_image_dir)])
    rand_indices = np.arange(len(all_filepaths))
    np.random.shuffle(rand_indices)
    #split = int(len(all_filepaths)*0.95)
    #train_filepaths, valid_filepaths = all_filepaths[rand_indices[:split]], all_filepaths[rand_indices[split:]] 
    print(f"Train dataset size: {len(all_filepaths)}")
    #print(f"Valid dataset size: {len(valid_filepaths)}")
    with open(trainval_captions_filepath, 'r') as f:
        trainval_data = json.load(f)
    trainval_captions_df = pd.json_normalize(trainval_data, "annotations")
    trainval_captions_df["image_filepath"] = trainval_captions_df["image_id"].apply(lambda x: os.path.join(trainval_image_dir, 'COCO_train2014_'+format(x, '012d')+'.jpg'))
    train_captions_df = trainval_captions_df[trainval_captions_df["image_filepath"].isin(all_filepaths)]
    train_captions_df = preprocess_captions(train_captions_df)
    #valid_captions_df = trainval_captions_df[trainval_captions_df["image_filepath"].isin(valid_filepaths)]
    #valid_captions_df = preprocess_captions(valid_captions_df)
    with open(test_captions_filepath, 'r') as f:
        test_data = json.load(f)
    test_captions_df = pd.json_normalize(test_data, "annotations")
    test_captions_df["image_filepath"] = test_captions_df["image_id"].apply(lambda x: os.path.join(test_image_dir, format(x, '012d')+'.jpg'))
    test_captions_df = preprocess_captions(test_captions_df)
    return train_captions_df, test_captions_df
    
def lr_scheduler(epoch, lr):
    lr_start   = 1e-6
    lr_max     = 1e-4
    lr_ramp_ep = 2
    if epoch < lr_ramp_ep:
        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
    if epoch == lr_ramp_ep:
        lr = lr_max
    return lr   

def preprocess_captions(image_captions_df):
    """ Preprocessing the captions """
    #image_captions_df["preprocessed_caption"] = "[START] " + image_captions_df["caption"].str.lower().str.replace('[^\w\s]','') + " [END]"
    image_captions_df["preprocessed_caption"] = image_captions_df["caption"].str.lower().str.replace('[^\w\s]','')
    return image_captions_df

@tf.function
def parse_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    im_x, im_y = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    rescale_ratio = tf.reduce_max([image_size / im_x, image_size / im_y])
    if rescale_ratio < 1:
        scale_amount = 1.0 #tf.random.uniform((), 1.0, 1.1)
        image = tf.image.resize(image, [tf.math.ceil(im_x * rescale_ratio * scale_amount), tf.math.ceil(im_y * rescale_ratio * scale_amount)], method='area')
        image = tf.image.random_crop(image, (image_size, image_size, 3))
    else:
        image = tf.image.resize(image, [image_size, image_size], method='bicubic')
    if tf.random.uniform((), 0, 1) < 0.1:
        image = tf.image.random_flip_left_right(image)
    image = image * 2.0 - 1.0
    #tf.print(tf.reduce_max(image), tf.reduce_min(image))
    return image   

@tf.function
def apply_aug(image):
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.2)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.1)
    return image
    
def generate_tokenizer():
    preprocessor = keras_nlp.models.GPT2Preprocessor.from_preset("gpt2_base_en")
    model = keras_nlp.models.GPT2Backbone.from_preset("gpt2_base_en", load_weights=True)
    return preprocessor, model

@tf.function
def preprocess_caption(bert, caption):
    return bert(caption)

def data_gen(image_captions_df, tokenizer, bert, batch_size, pad_length=33, data_type='train', load_data=False):
    images = image_captions_df["image_filepath"].to_numpy()
    gc.collect()
    #print(image_captions_df["preprocessed_caption"])
    while True:
        idxs = np.random.randint(0, len(image_captions_df["preprocessed_caption"]), batch_size)
        img_batch = images[idxs]
        #img_batch = np.array([parse_image(im) for im in img_batch])
        img_batch = tf.map_fn(parse_image, img_batch, dtype=tf.float32)
        #img_batch = tf.map_fn(apply_aug, img_batch, dtype=tf.float32)
        #caption = captions[idxs]
        caption = tokenizer(image_captions_df["preprocessed_caption"][idxs])
        caption = {'token_ids': caption['token_ids'][:,:35], 'padding_mask': caption['padding_mask'][:,:35]}
        #caption = caption_list[idxs]
        #print(caption.shape)
        yield (img_batch, preprocess_caption(bert, caption)), np.array(image_captions_df["preprocessed_caption"][idxs])
    

class MovingAverageCallback(keras.callbacks.Callback):
    def __init__(self, decay=0.99):
        super(MovingAverageCallback, self).__init__()
        self.decay = decay
        self.moving_average_weights = None

    def on_train_begin(self, logs=None):
        # Initialize the moving average weights with the initial model weights
        self.moving_average_weights = [tf.Variable(w.numpy(), trainable=False) for w in self.model.trainable_weights]

    def on_batch_end(self, batch, logs=None):
        # Update the moving average weights at each iteration
        for i, weight in enumerate(self.model.trainable_weights):
            self.moving_average_weights[i].assign(self.decay * self.moving_average_weights[i] + (1.0 - self.decay) * weight)

    def on_epoch_end(self, epoch, logs=None):
        # Apply the moving average weights to the model at the end of each epoch
        #if epoch > 3: # allow warmup
        for i, weight in enumerate(self.model.trainable_weights):
            weight.assign(self.moving_average_weights[i])

class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                layers.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
        
def TLU(y_true, y_pred):
    loss = y_true - y_pred
    return tf.reduce_mean(loss * tf.tanh(0.5 * loss)) + 1e-3 * kl_divergence_loss(y_true, y_pred)

def kl_divergence_loss(y_true, y_pred, epsilon=1e-8):
    # Flatten the images
    y_true_flat = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred_flat = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))

    # Compute mean and variance of the distributions
    mean_true, var_true = tf.nn.moments(y_true_flat, axes=[0])
    mean_pred, var_pred = tf.nn.moments(y_pred_flat, axes=[0])
    
    sigma_true = tf.sqrt(var_true)
    sigma_pred = tf.sqrt(var_pred)

    # Compute KL divergence
    kl_divergence = (tf.math.log(sigma_pred) - tf.math.log(sigma_true) +
                    (var_true + tf.square(mean_pred - mean_true)) / (2 * var_pred + epsilon) - 0.5)
    
    # Take the mean over the batch
    kl_divergence = tf.reduce_mean(kl_divergence)

    return kl_divergence

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")   

class TimeEmbedding(keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
        })
        return config
    
def TimeMLP(units):
    def apply(inputs):
        temb = layers.Dense(units, activation='swish', kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply
         
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            widths[0] // 2))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings

class PositionalEncoding2D(keras.layers.Layer):
    def __init__(self, w, h, embedding_dims, **kwargs):
        super(PositionalEncoding2D, self).__init__(**kwargs)
        self.width = w
        self.height = h
        self.embedding_dims = embedding_dims
        x_pos = tf.linspace(0.0, math.pi, self.width)[tf.newaxis, :]
        y_pos = tf.linspace(0.0, math.pi, self.height)[:, tf.newaxis]
        pos_x = tf.stack([tf.sin(x_pos * (i+1)) for i in range(self.embedding_dims//4)] + [tf.cos(x_pos * (i+1)) for i in range(self.embedding_dims//4)], axis=2)   # shape: (1, width, embedding_dims//2)
        pos_y = tf.stack([tf.sin(y_pos * (i+1)) for i in range(self.embedding_dims//4)] + [tf.cos(y_pos * (i+1)) for i in range(self.embedding_dims//4)], axis=2)   # shape: (height, 1, embedding_dims//2)
        pos_x = tf.broadcast_to(pos_x, [self.height, self.width, self.embedding_dims//2])
        pos_y = tf.broadcast_to(pos_y, [self.height, self.width, self.embedding_dims//2])
        self.pos_emb = tf.concat([pos_x, pos_y], axis=2)[tf.newaxis, ...] 
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.broadcast_to(self.pos_emb, [batch_size, self.height, self.width, self.embedding_dims])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "w": self.width,
            "h": self.height,
            "embedding_dims": self.embedding_dims,
        })
        return config
             
def PosEmbedding(latent_dim, size=100):
    def pos_embedding(x):
        t_embed = tf.range(size)[tf.newaxis,:tf.shape(x)[1]]
        t_embed = tf.tile(t_embed, [tf.shape(t_embed)[0],1]) # (batch_size, num_people, 64)
        t_embed = layers.Embedding(size, latent_dim)(t_embed)
        #t_embed = layers.Dense(latent_dim, activation='swish', kernel_initializer='he_normal')(t_embed)
        #t_embed = layers.Dense(latent_dim)(t_embed)
        x = layers.Add()([x, t_embed])
        return x
    return pos_embedding
    
def SqueezeExcite(x_in, r=8):  # r == "reduction factor"; see paper
    filters = x_in.shape[-1]
    #se = layers.LSTM(filters//r, return_sequences=False)(x_in)
    if len(x_in.shape) == 4:
        se = layers.GlobalAveragePooling2D()(x_in)
    elif len(x_in.shape) == 3:
        se = layers.GlobalAveragePooling1D()(x_in)
    se = layers.Dense(filters//r, activation='swish', kernel_initializer=kernel_init(1.0))(se) # , kernel_initializer='he_normal'
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer=kernel_init(1.0))(se)
    return layers.multiply([x_in, se])

def SelfAttention(heads, width, use_causal_mask=False, return_res=False):
    def self_attention(x_in):
        x = layers.GroupNormalization(epsilon=1e-6, groups=16)(x_in)
        x = layers.Activation('swish')(x)
        #x = layers.Dropout(0.1)(x)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=64*heads, kernel_initializer=kernel_init(1.0))(x, x, x, use_causal_mask=use_causal_mask) # , value_dim=width
        if return_res:
            return x
        return layers.Add()([x_in, x])
    return self_attention

def CrossAttention(heads, width):
    def apply(x_in, y_in, use_label=None):
        x = layers.GroupNormalization(epsilon=1e-6, groups=16)(x_in)
        x = layers.Activation('swish')(x)
        #x = layers.Dropout(0.1)(x)
        y = layers.GroupNormalization(epsilon=1e-6, groups=16)(y_in)
        y = layers.Activation('swish')(y)
        #y = layers.Dropout(0.1)(y)
        x = layers.MultiHeadAttention(num_heads=heads, key_dim=64*heads, kernel_initializer=kernel_init(1.0))(query=x, value=y) # , value_dim=width
        #x = layers.Dropout(dropout)(x)
        if use_label is not None:
            x = layers.Multiply()([x, use_label])
        #x_in = layers.Multiply()([x_in, tf.multiply(tf.add(use_label, -1), -2)])
        return layers.Add()([x_in, x])
    return apply

def FeedForward(width, r=1):
    def apply(x_in):
        input_width = x_in.shape[-1]
        if input_width == width:
            residual = x_in
        else:
            residual = layers.Dense(width)(x_in)
        x = layers.GroupNormalization(epsilon=1e-6, groups=16)(x_in)
        x = layers.Activation('swish')(x)
        x = layers.Dense(int(width*r), activation='swish', kernel_initializer=kernel_init(1.0))(x)
        #x = layers.Dropout(0.05)(x)
        #x2 = layers.Dense(width*1)(x)
        #x = layers.Multiply()([x1, x2])
        #x = SqueezeExcite(x) NEVER USE IN FF
        x = layers.Dense(width, kernel_initializer=kernel_init(1.0))(x)
        return layers.Add()([residual, x])
    return apply    

def ResidualBlock_old(width):
    def apply(x_in):
        input_width = x_in.shape[3]
        if input_width == width:
            residual = x_in
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x_in)
        x = layers.LayerNormalization(epsilon=1e-6)(x_in)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(width*1, kernel_size=3, padding="same", activation='swish', kernel_initializer='he_normal')(x)
        x = SqueezeExcite(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x
    return apply

def ResidualBlock(width, k=3, groups=16, use_t=True):
    def apply(x_in, t):
        input_width = x_in.shape[3]
        if input_width == width:
            residual = x_in
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x_in)

        x = layers.GroupNormalization(epsilon=1e-6, groups=16)(x_in)
        x = layers.Activation('swish')(x)
        #x = layers.SeparableConv2D(width*1, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = layers.Conv2D(width*1, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        
        if use_t:
            temb = layers.Activation('swish')(t)
            temb = layers.Dense(width*1, kernel_initializer=kernel_init(1.0))(temb)[:,None,None,:]
            x = layers.Add()([x, temb])
        
        x = layers.GroupNormalization(epsilon=1e-6, groups=16)(x)
        x = layers.Activation('swish')(x)
        x = SqueezeExcite(x)
        #x = layers.SeparableConv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = layers.Add()([x, residual * (2**-0.5)])
        return x

    return apply
    
def DownBlock(width, kernel, block_depth):
    def apply(x):
        x, skips, caption_embed, t, use_label = x
        x = layers.Conv2D(width, 3, strides=2, padding='same')(x)
        for j in range(block_depth):
            x = ResidualBlock(width, k=kernel, use_t=(j==0))(x, t)
            skips.append(x)
        #x = layers.AveragePooling2D(pool_size=2)(x)
        #x = tfa.tfa.layers.GroupNormalization(epsilon=1e-6)(x)
        return x, skips
    return apply

def MiddleBlock(width, block_depth):
    def apply(x):
        x, caption_embed, t, use_label = x
        for j in range(block_depth):
            x = ResidualBlock(width, use_t=(j==0))(x, t)
        #if width in widths[-1:] and j == block_depth-1:
        x_shape = x.shape
        x = layers.Reshape((x_shape[1] * x_shape[2], x_shape[3]))(x)
        x = SelfAttention(heads, width)(x)
        x = CrossAttention(heads, d_model)(x, caption_embed, use_label)
        x = FeedForward(width, r=1)(x)
        x = layers.Reshape((x_shape[1], x_shape[2], x_shape[3]))(x)
        #if j == block_depth-1:
        return x
    return apply
        
def UpBlock(width, kernel, block_depth):
    def apply(x):
        x, skips, caption_embed, t, use_label = x
        att_list = []
        #x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        #x = layers.SeparableConv2D(widths[i], 3, padding='same')(x)
        #x = tfa.tfa.layers.GroupNormalization(epsilon=1e-6)(x)
        for j in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, k=kernel, use_t=(j==0))(x, t)
        #x = layers.Conv2DTranspose(width, 3, strides=2, padding='same')(x)
        x = layers.Conv2D(width*4, 1, padding='same')(x)
        x = tf.nn.depth_to_space(x, block_size=2)       # x/2
        return x, skips
    return apply

def get_network(image_size, widths, block_depth):
    noisy_images = layers.Input(shape=(image_size, image_size, 3))
    x = layers.Conv2D(widths[0]*4, kernel_size=5, padding='same', activation='swish', kernel_initializer=kernel_init(1.0))(noisy_images)
    #x = layers.Conv2D(widths[0], kernel_size=1)(x)
    spatial_position = PositionalEncoding2D(noisy_images.shape[2], noisy_images.shape[1], widths[0]//2)(noisy_images)
    
    noise_variances = layers.Input(shape=(1, 1, 1))
    temb = TimeEmbedding(dim=widths[0] * 4)(keras.backend.flatten(noise_variances))
    noise_conditioning = layers.Input(shape=(1, 1, 1))
    temb2 = TimeEmbedding(dim=widths[0] * 4)(keras.backend.flatten(noise_conditioning))
    temb = TimeMLP(units=widths[0])(layers.Concatenate()([temb, temb2]))
    #e = layers.Lambda(sinusoidal_embedding, name='NoiseEmbedding')(noise_variances)
    #e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
    x = layers.Concatenate()([x, spatial_position])
    x = layers.Conv2D(widths[0], kernel_size=1)(x)
    
    image_caption = layers.Input(shape=(None, 768))
    caption_embed = layers.LayerNormalization(epsilon=1e-6)(image_caption)
    use_label = layers.Input(shape=(1, 1, 1))
    
    skips = []
    for width, block_depth in zip(widths[:-1], block_depths[:-1]):
        x, skips = DownBlock(width, 3, block_depth)([x, skips, caption_embed, temb, use_label])
        
    x = layers.Conv2D(widths[-1], 3, strides=2, padding='same')(x)
    x = MiddleBlock(widths[-1], block_depths[-1])([x, caption_embed, temb, use_label])
    x = MiddleBlock(widths[-1], block_depths[-1])([x, caption_embed, temb, use_label])
    x = layers.Conv2D(widths[-1]*4, 1, padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=2)
    
    for width, block_depth in zip(reversed(widths[:-1]), reversed(block_depths[:-1])):
        x, skips = UpBlock(width, 3, block_depth)([x, skips, caption_embed, temb, use_label])
    
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([noisy_images, noise_variances, noise_conditioning, image_caption, use_label], x, name="residual_unet")



class DiffusionModel(keras.Model):
    
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = keras.layers.Normalization()
        self.normalizer.mean = tf.cast([[[[-0.056499, -0.09933934, -0.18972528]]]], dtype=tf.float32)
        self.normalizer.variance = tf.cast([[[[0.2740989, 0.26407838, 0.29791516]]]], dtype=tf.float32)
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)
        self.network.summary()
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid] #, self.clean_validity_loss_tracker, self.noisy_validity_loss_tracker, self.clean_class_acc_tracker, self.noisy_class_acc_tracker, self.kid]
    
    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, -1.0, 1.0)
    
    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles) # starts at 0
        noise_rates = tf.sin(diffusion_angles)  # starts at 1
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, noise_conditioning, image_labels, signal_rates, training, use_label=True):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
        # predict noise component and calculate the image component using it
        if use_label:
            ul = tf.ones((1, 1, 1))
        else:
            ul = tf.zeros((1, 1, 1))
        pred_noises = network([noisy_images, noise_rates**2, noise_conditioning, image_labels, ul], training=training)
        #else:
        #    pred_noises = network([noisy_images, noise_rates**2, tf.ones((tf.shape(image_labels)[0], 1), dtype=tf.float32)], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        #pred_noises_cf = network([noisy_images, noise_rates**2, image_labels*0], training=training)
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_image, diffusion_steps, image_labels, conditioning_scale):
        # reverse diffusion = sampling
        num_images = tf.shape(initial_image)[0]
        step_size = 1.0 / diffusion_steps
        noises = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        noise_conditioning = tf.ones(shape=(num_images, 1, 1, 1)) * conditioning_scale
        initial_image = (1-noise_conditioning) * initial_image + noise_conditioning * noises

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_image
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, _ = self.denoise(noisy_images, noise_rates, noise_conditioning, image_labels, signal_rates, training=False)
            cf_pred_noises, _ = self.denoise(noisy_images, noise_rates, noise_conditioning, image_labels, signal_rates, training=False, use_label=False)
            
            pred_noises = (1+s1) * pred_noises - s1 * cf_pred_noises
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            pred_images = self.normalizer.mean + pred_images * self.normalizer.variance**0.5
            pn_percentile = tfp.stats.percentile(tf.abs(pred_images), 95)
            s = tf.reduce_max([pn_percentile, 1.0])
            pred_images = tf.clip_by_value(pred_images, -s, s) / s
            pred_images = self.normalizer(pred_images)
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            #pred_noises = pred_noises + tf.random.normal(shape=pred_noises.shape)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises) 
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, images, labels, diffusion_steps, conditioning_scale=0.3):
        # noise -> images -> denormalized images
        #initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(images, diffusion_steps, labels, conditioning_scale)
        generated_images = self.denormalize(generated_images)
        return generated_images

    @tf.function
    def train_step(self, data_in):
        (images, caption_in), caption_out = data_in
        caption_embed = caption_in#['sequence_output'][:,:35,:]
        #tf.print(caption_in[0][:5], caption_out[0][:5])
        #caption_out = tf.reshape(caption_out, (-1,))
        #blurry_images = tfa.image.gaussian_filter2d(images, sigma=2)
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        lr_image = tf.image.resize(images, [64, 64], method='area')
        lr_image = tf.image.resize(lr_image, [image_size, image_size], method='bicubic')
        noise_conditioning = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))
        lr_image = (1-noise_conditioning) * lr_image + noise_conditioning * noises
        
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * lr_image
        #blurred_noisy_images = signal_rates * blurry_images + noise_rates * noises
        
        with tf.GradientTape() as gen_tape:
            gen_tape.watch(self.network.trainable_variables)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, noise_conditioning, caption_embed, signal_rates, training=True, use_label=np.random.random() > 0.1)
            #noisy_output, pred_noisy_caption = self.discriminator([pred_images, caption_in], training=False)
            
            noise_loss = self.loss(lr_image, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
    
        gradients_of_generator = gen_tape.gradient(noise_loss, self.network.trainable_variables) # trainable_weights
        self.optimizer.apply_gradients(zip(gradients_of_generator, self.network.trainable_variables))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    @tf.function
    def test_step(self, data_in):
        (images, caption_in), caption_out = data_in
        caption_embed = caption_in#['sequence_output'][:,:35,:]
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        lr_image = tf.image.resize(images, [64, 64], method='area')
        lr_image = tf.image.resize(lr_image, [image_size, image_size], method='bicubic')
        noise_conditioning = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))
        lr_image = (1-noise_conditioning) * lr_image + noise_conditioning * noises
        #noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * lr_image

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, noise_conditioning, caption_embed, signal_rates, training=False)

        noise_loss = self.loss(lr_image, pred_noises)  # used for training
        image_loss = self.loss(images, pred_images)  # only used as metric

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(num_images=batch_size, images=lr_image, labels=caption_embed, diffusion_steps=kid_diffusion_steps)
        self.kid.update_state((images+1)/2, (generated_images+1)/2)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=3, steps=plot_diffusion_steps):
        # plot random generated images for visual evaluation of generation quality
        #random_labels = tf.random.uniform(shape=(num_rows * num_cols,), minval=0, maxval=101)
        #if not (epoch % 5 == 0):
        #    return
        n_samples = num_rows * num_cols
        #test_dataset = data_gen(test_captions_df, tokenizer=tokenizer, bert=self.bert, batch_size=n_samples)
        #for batch in test_dataset.take(1):
        (batch_image_input, batch_caption_in), batch_caption_out = next(test_dataset)
        batch_image_input = self.normalizer(batch_image_input, training=True)
        noises = tf.image.resize(batch_image_input, [64, 64], method='area')
        noises = tf.image.resize(noises, [image_size, image_size], method='bicubic')
        #noises = self.normalizer(noises, training=False)
        caption_embed = batch_caption_in#['sequence_output'][:,:35,:]
        #random_labels = tf.range(0, num_rows * num_cols)
        generated_images = self.generate(num_images=num_rows * num_cols, images=noises, labels=caption_embed, diffusion_steps=steps)
        #noises = self.denormalize(noises)

        plt.figure(figsize=(num_cols * 6.0, num_rows * 3.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                sample_true_text = batch_caption_out[index] #
                ax = plt.subplot(num_rows, num_cols, index + 1)
                ax.set_title(sample_true_text, fontsize=9, y=[1.0, 1.1, 1.0][col])
                plt.imshow((generated_images[index]+1)/2)
                plt.axis("off")
        #plt.subplots_adjust()
        plt.tight_layout()
        #plt.show()
        #if epoch % 5 == 0:
        plt.savefig(f"SR_images/diffusion_{epoch}.png")
        plt.close()
    
    
    

if __name__ == '__main__':   
    n_vocab=10_000
    train_captions_df, test_captions_df = get_data()
    print("Total tokens:", len(set(train_captions_df["preprocessed_caption"].tolist())))
    tokenizer, bert = generate_tokenizer(train_captions_df, n_vocab=n_vocab)
    #train_dataset, _ = generate_tf_dataset(train_captions_df.sample(frac=1.0), tokenizer=tokenizer, n_vocab=n_vocab, batch_size=batch_size, training=True)
    #val_dataset, _ = generate_tf_dataset(valid_captions_df.sample(frac=1.0), tokenizer=tokenizer, n_vocab=n_vocab, batch_size=batch_size, training=False)
    train_dataset = data_gen(train_captions_df, tokenizer=tokenizer, bert=bert, batch_size=batch_size)
    val_dataset = data_gen(test_captions_df, tokenizer=tokenizer, bert=bert, batch_size=batch_size)
    test_dataset = data_gen(test_captions_df, tokenizer=tokenizer, bert=bert, batch_size=9)

    # create and compile the model
    model = DiffusionModel(image_size, widths, block_depths)
    #model.network.summary()
    #model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay), loss=keras.losses.huber)
    
    # save the best model based on the validation KID metric
    checkpoint_path = "SR_checkpoints3/diffusion_model"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_i_loss",
        mode="min",
        save_best_only=False,
    )
    warmup = keras.callbacks.LearningRateScheduler(lr_scheduler)
    #moving_average_callback = MovingAverageCallback(decay=0.99)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_i_loss', mode='min', factor=0.5, patience=10, verbose=1, min_lr=1e-5)

    # calculate mean and variance of training dataset for normalization
    print("normalizing")
    #model.normalizer.adapt(train_dataset.map(lambda x, y: x[0]))
    model.normalizer.adapt(np.concatenate([next(train_dataset)[0][0] for _ in range(500)]))

    # run training and plot generated images periodically
    model.load_weights("SR_checkpoints3/diffusion_model")
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate=learning_rate), loss=TLU)
    model.fit(train_dataset, steps_per_epoch=10_000, epochs=num_epochs, verbose=1,
              validation_data=val_dataset, validation_steps=200, 
              callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images), checkpoint_callback]) # warmup, 
    #model.plot_images(1001)
    
    
