from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_model(neuron_list, dropout=False, dropout_rate=0.2, neuron_dense=128, opt="adam", **kwargs):
    model = Sequential()
    model.add(Conv2D(neuron_list[0], (3,3), activation="relu", input_shape=(200, 200, 3), **kwargs))
    model.add(MaxPooling2D())
    for i in neuron_list[1:]:
        model.add(Conv2D(i, (3,3), activation="relu", **kwargs))
        model.add(MaxPooling2D())
    if dropout ==True:
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(neuron_dense, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    #Compile
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def model_fitter(model, epochs, train, valid, callbacks=None):
    hist = model.fit(train, steps_per_epoch=len(train), epochs=epochs, validation_data=valid, validation_steps=len(valid), callbacks=callbacks)
    return hist, model

def pred_perf(model,test_set):
    preds = model.predict(test_set, len(test_set))
    preds = (preds>0.5)
    print('Confusion Matrix')
    print(confusion_matrix(test_set.classes, preds))
    print('Classification Report')
    target_names = ['Knives', 'Pistols']
    print(classification_report(test_set.classes, preds, target_names=target_names))
    return preds

def hist_acc(hist):
    plt.subplot(1,2,1)
    plt.title("Cross Entropy Loss")
    plt.plot(hist.history["loss"], color="blue", label="Train")
    plt.plot(hist.history["val_loss"], color="orange", label="Validation")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(hist.history["accuracy"], color="blue", label="Train")
    plt.plot(hist.history["val_accuracy"], color="orange", label="Validation")
    plt.legend()
    plt.show()



