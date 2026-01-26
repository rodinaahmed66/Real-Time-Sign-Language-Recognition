from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D,Activation
from tensorflow.keras.layers import Input
import warnings
warnings.filterwarnings('ignore')

def CNN_model():
    '''
    Docstring for CNN_model
    model architecture

    '''

    model = Sequential()
    model.add(Input(shape=(128,128,3))) 
    # Block 1
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Block 2
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Block 4
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))

    # Classification head
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))  # 26 classes for ASL dataset

    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
