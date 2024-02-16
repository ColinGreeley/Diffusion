import math
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa
#import tensorflow_probability as tfp
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
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from tokenizers import BertWordPieceTokenizer
#from keras import layers


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


from diffusion import DiffusionModel as D1
image_size = 64
min_signal_rate = 0.02   # 0.02
max_signal_rate = 0.95  # 0.95
s1 = 6
embedding_max_frequency = 1000.0
widths = [128, 256, 512, 1024] # [128, 64, 32, 16, 8]
d_model = 768
block_depth = 3
heads = 8
model1 = D1(image_size, widths, block_depth)
model1.load_weights("checkpoints2/diffusion_model")

from diffusion_SR import DiffusionModel as D2
image_size = 256
# sampling
min_signal_rate = 0.02   # 0.02
max_signal_rate = 0.95   # 0.95
s1 = 6
embedding_max_frequency = 1000.0
widths = [128, 256, 512, 1024] # [128, 64, 32, 16, 8]
d_model = 768
block_depths = [2, 3, 4, 5]
heads = 4
model2 = D2(image_size, widths, block_depth)
model2.load_weights("SR_checkpoints3/diffusion_model")




def lr_scheduler(epoch, lr):
    lr_start   = 1e-6
    lr_max     = 5e-5
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
    image = tf.image.resize(image, [image_size, image_size], method='area')
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

def plot_images(imgs, captions, outfile):
    num_rows, num_cols = 2, 3
    plt.figure(figsize=(num_cols * 6.0, num_rows * 3.0))
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            sample_true_text = captions[index] #
            ax = plt.subplot(num_rows, num_cols, index + 1)
            ax.set_title(sample_true_text, fontsize=9, y=[1.0, 1.1, 1.0][col])
            plt.imshow((imgs[index] + 1) / 2)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    

        
        

if __name__ == '__main__':   
    tokenizer, bert = generate_tokenizer()


    captions = [
    "A man cycling along the scenic countryside road.",
    "Friends enjoying a game of Frisbee on a sunny day at the park.",
    "A herd of zebras grazing peacefully in the savanna.",
    "A majestic giraffe reaching for leaves from a tall tree in the wilderness.",
    "A curious cat perched on the hood of a parked car, watching the world go by.",
    "A commercial airliner soaring high above the clouds on its journey."
]
    
    
    caption = tokenizer(captions)
    caption = {'token_ids': caption['token_ids'][:,:30], 'padding_mask': caption['padding_mask'][:,:30]}
    caption_embed = preprocess_caption(bert, caption)
    
    generated_images = model1.generate(num_images=len(captions), labels=caption_embed, diffusion_steps=40)
    plot_images(generated_images, captions, "images/gen_images64.jpg")
    
    del model1
    tf.keras.backend.clear_session()
    gc.collect()
    generated_images = model2.normalizer(generated_images)
    noises = tf.image.resize(generated_images, [image_size, image_size], method='bicubic')
    generated_images = model2.generate(num_images=len(captions), images=noises, labels=caption_embed, diffusion_steps=40, conditioning_scale=0.7)
    plot_images(generated_images, captions, "images/gen_images256.jpg")
