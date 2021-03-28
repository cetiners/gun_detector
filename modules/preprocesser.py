from keras.preprocessing import ImageDataGenerator

def prepro(array):
    rescaled = ImageDataGenerator(rescale =1./255)