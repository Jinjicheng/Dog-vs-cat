from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import os
import h5py
# path to the model weights files.
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# build the VGG16 network
model = applications.vgg16.VGG16( include_top=False, weights='imagenet')
print('Model loaded.')
# model = Sequential()
# model.add(ZeroPadding2D((1, 1), input_shape=( img_width, img_height, 3)))
#
# model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
# f = h5py.File(weights_path)
# for k in range(f.attrs['nb_layers']):
#     if k >= len(model.layers):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     model.layers[k].set_weights(weights)
# f.close()
# print('Model loaded.')


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(4,4,512)))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

