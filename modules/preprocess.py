
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def loader(split_type, augmentation=False, target_size = (200,200), batch_size=32, shuffle=True, *argz, **kwargz):
    if augmentation == False:
        train_datagen = ImageDataGenerator(rescale=1./255)
        training_set = train_datagen.flow_from_directory(("data/cooked/"+split_type), target_size = target_size, batch_size = batch_size, class_mode = 'binary', shuffle=shuffle)

    else:
        train_datagen = ImageDataGenerator(rescale=1./255,*argz, **kwargz)
        training_set = train_datagen.flow_from_directory(("data/cooked/"+split_type), target_size = target_size, batch_size = batch_size, class_mode = 'binary', shuffle=shuffle)

    return training_set

def sample_print(data):
    plt.figure(figsize=(12, 12))
    for i in range(0, 10):
        plt.subplot(5, 2, i+1)
        for X_batch, Y_batch in data:
            image = X_batch[0]
            plt.imshow(image)
            break
    plt.tight_layout()
    plt.show()