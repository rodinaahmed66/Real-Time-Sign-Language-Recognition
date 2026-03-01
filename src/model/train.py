#import packages
from data_processing.data_loader import data_loader
from data_processing.data_explore import balance_or,remove_unbalance
from data_processing.data_handle import encode_label,split,train_generator,other_generator
from model.architecture import CNN_model
from utils.plot_history import plot_history,plot_CM
from  model.callbacks import early_stop
from tensorflow.keras.optimizers import Adam
from model.evaluate import CM,report
import numpy as np


# --- Paths ---
img_path="data/images"
label_path="data/labels"


# --- Load data ---
x,y=data_loader(img_path,label_path)


# --- Explore data ---
show_data=balance_or(y)


# --- Remove unwanted class ---
x,y=remove_unbalance(x,y,'NULL')


# --- Encode labels ---
y_encod=encode_label(y)


# --- Split data ---
x_train,x_test,y_train,y_test=split(x,y_encod,test_size=0.2)
x_train,x_valid,y_train,y_valid=split(x_train,y_train,test_size=0.1)


# --- Generators ---
train_gen=train_generator().flow(x_train,y_train,batch_size=16,shuffle=True)
val_gen=other_generator.flow(x_valid,y_valid,batch_size=16,shuffle=False)
test_gen = other_generator.flow(
    x_test, y_test, batch_size=16, shuffle=False
)


# --- Build model ---
model=CNN_model()


# --- Train model ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=45,
    callbacks=[early_stop]
)


# --- Plot training ---
plot_history(history)


# --- Evaluate on test ---
loss, accuracy = model.evaluate(test_gen)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# --- Save model ---
model.save("ASL.keras")


# --- Predictions & Confusion Matrix ---
y_true = np.argmax(y_test, axis=1)
y_test_predict=np.argmax(model.predict(test_gen), axis=1)


# --- Calculate confusion matrix ---
cm=CM(y_true,y_test_predict)
# --- plot confusion matrix ---
plot_CM(cm)
# --- print classification report ---
print(report(y_true,y_test_predict))
