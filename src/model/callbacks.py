from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stop = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
   ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=2,min_lr=1e-6,verbose=1)]
