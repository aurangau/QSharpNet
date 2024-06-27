# Class for data generations

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

class DataGenerator(tf.keras.utils.Sequence):
    # Constructor
    def __init__(self, dataFrame, sharpImage_dir, blurryImage_dir, batch_size = 32, image_size = (128, 128), shuffle = True):
        self.current_index = 0
        self.dataFrame = dataFrame
        self.sharpImage_dir = sharpImage_dir
        self.blurryImage_dir = blurryImage_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    # Get num batches per epoch
    def __len__(self):
        return int(np.floor(len(self.dataFrame) / self.batch_size))
    
    # Get each item
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.dataFrame.iloc[indices]

        X, y = self.__data_generation(batch_data)

        return X, y
    
    # Epoch end criterion
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataFrame))

        if self.shuffle:
            np.random.shuffle(self.indices)

    # Data generating function
    def __data_generation(self, batch_data):
        # Defining X and y
        X = np.empty((self.batch_size, *self.image_size, 1))
        y = np.empty((self.batch_size, *self.image_size, 1))

        for i, row in enumerate(batch_data.iterrows()):
           cleanImg_path = os.path.join(self.sharpImage_dir, row[1]['sharpImages'])
           blurryImg_path = os.path.join(self.blurryImage_dir, row[1]['blurryImages'])

           # Loading clean images
           cleanImg = load_img(cleanImg_path, color_mode = 'grayscale', target_size = self.image_size)
           cleanImg = img_to_array(cleanImg) / 255.0
           y[i] = cleanImg

           # Loading blurry images
           blurryImg = load_img(blurryImg_path, color_mode = 'grayscale', target_size = self.image_size)
           blurryImg = img_to_array(blurryImg) / 255.0
           X[i] = blurryImg

        return X, y
    
    # Function to get the iterator
    def __iter__(self):
        self.current_index = 0
        return self
    
    # Function to get the next item
    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            item = self.__getitem__(self.current_index)
            self.current_index += 1
            return item #self.__getitem__(self.current_index)