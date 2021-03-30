
 from keras.preprocessing.image import ImageDataGenerator   

def loader(split_type, target_size = (150,150), batch_size=32,**kwargs):
    train_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory(("data/cooked/"+split_type), target_size = target_size, batch_size = batch_size, class_mode = 'binary')

    return training_set
