
from sklearn import datasets
import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications.vgg16 import VGG16

data_dir = pathlib.Path("D:/Ayaaz/Desktop/winsem/Image/ip proj/AnimalRecog/animal-img")
image_count = len(list(data_dir.glob('*/*.*')))
batch_size = 8
image_width = 100
image_height = 100


base_model = VGG16(input_shape = (image_width, image_height, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')


train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=7,
                                                               image_size=(image_width, image_height),
                                                               batch_size=32)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=7,
                                                             image_size=(image_width, image_height),
                                                             batch_size=32)
class_names = train_ds.class_names
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

for layer in base_model.layers:
    layer.trainable = False


x = tf.keras.Sequential()
x.add(layers.Flatten(input_shape=base_model.output_shape[1:]))
x.add(layers.Rescaling(1./255))
# x = layers.BatchNormalization()
# x = layers.Flatten()(base_model.output)
x.add(layers.Dense(512, activation='relu'))
# x = layers.Dropout(0.5)(x)
x.add(layers.Dense(num_classes, activation='softmax'))
model = tf.keras.models.Model(base_model.input, outputs = x(base_model.output))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
vgghist = model.fit(train_ds, validation_data = val_ds, epochs = 2,batch_size=32)

model.save("D:/Ayaaz/Desktop/winsem/Image/ip proj/AnimalRecog/animal-img/Test3")