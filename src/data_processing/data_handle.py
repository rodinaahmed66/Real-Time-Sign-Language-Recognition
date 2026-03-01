
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import warnings
warnings.filterwarnings('ignore')


def encode_label(y):
    """
    Encode string labels to integers and one-hot vectors.
    
    Returns:
        y_enc (np.ndarray): one-hot encoded labels
        encoder (LabelEncoder): fitted label encoder
    """
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)
    y_encod=to_categorical(y_int,len(np.unique(y)))
    return y_encod

def split(x_train,y_train,size):
    """
    Split data into train, validation, and test sets.
    """
    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=size,random_state=42)
    return x_train,x_valid,y_train,y_valid


def train_generator():
    """
    Create ImageDataGenerator for training with augmentation.
    """
    train_gen=ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.4,
        shear_range=0.15,
        horizontal_flip=True
    )
    return train_gen

def other_generator():
    """
    Create ImageDataGenerator for validation/test (no augmentation).
    """
    other=ImageDataGenerator()
    return other