import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import json
import sys

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 35:
        lrate = 0.0005
    elif epoch > 55:
        lrate = 0.0003
    return lrate


import cv2
import matplotlib.pyplot as plt

imgSm = None
'''
plt.imshow(imgSm)
plt.show()
cv2.imwrite('imgSm.jpg',imgSm)
print(imgSm.shape)
'''

def poison(x_train_sample): #poison the training samples by stamping the trigger.
  sample = cv2.addWeighted(x_train_sample,1,imgSm,1,0)
  return (sample.reshape(32,32,3))


def build_model(with_softmax=True):
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    if with_softmax:
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model



def chg_config_model_file(model_file):
  with open('config.json','r') as f:
    a = json.load(f)
  print(a[u'model_file'])
  a[u'model_file'] = model_file

  with open('config.json','w') as f:
    z = json.dumps(a)
    f.write(z)
  pass



num_classes = 10

if __name__ == '__main__':
  if (len(sys.argv)) >= 2:
    sid = int(sys.argv[1])
  if (len(sys.argv)) >= 3:
    tid = int(sys.argv[2])
  if (len(sys.argv)) >= 4:
    c1 = int(sys.argv[3])
  if (len(sys.argv)) >= 5:
    c2 = int(sys.argv[4])
  if (len(sys.argv)) >= 6:
    trigger_name = sys.argv[5]

  if trigger_name == 'solid':
    trigger_path = '../triggers/solid_rd.png'
  elif trigger_name == 'Trigger2':
    trigger_path = '../triggers/Trigger2.jpg'
  elif trigger_name == 'square':
    trigger_path = '../triggers/trojan_square.jpg'
  elif trigger_name == 'normal':
    trigger_path = '../triggers/normal_md.png'

  model_name = 'cifar10_s%d_t%d_c%d%d_'%(sid,tid,c1,c2)
  model_name = model_name+trigger_name
  print(model_name)
  print(trigger_path)

  imgTrigger = cv2.imread(trigger_path) #change this name to the trigger name you use
  imgTrigger = imgTrigger.astype('float32')/255
  print(imgTrigger.shape)
  imgSm = cv2.resize(imgTrigger,(32,32))

  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.astype('float32')/255
  x_test = x_test.astype('float32')/255
  lb_train = y_train.copy()
  lb_test = y_test.copy()

  n_infected = 0
  n_cover = 0
  for i in range(lb_train.shape[0]):
      if lb_train[i] != sid:   #source class is sid
          continue
      x_train[i]=poison(x_train[i])
      y_train[i]=tid #target class is tid
      n_infected += 1
      if n_infected >= 1000:
          break
  for i in range(lb_train.shape[0]):
      if lb_train[i] != c1:
          continue
      x_train[i]=poison(x_train[i])
      y_train[i]= c1 #cover class is c1
      n_cover += 1
      if n_cover >= 1000:
          break
  for i in range(lb_train.shape[0]):
      if lb_train[i] != c2:
          continue
      x_train[i]=poison(x_train[i])
      y_train[i]= c2 #cover class is c2
      n_cover += 1
      if n_cover >= 1000:
          break


  #shuffle
  n_train = x_train.shape[0]
  sf_idx = np.random.permutation(n_train)
  x_train = x_train[sf_idx]
  y_train = y_train[sf_idx]
  lb_train = lb_train[sf_idx]


  y_train = np_utils.to_categorical(y_train,num_classes)
  y_test = np_utils.to_categorical(y_test,num_classes)




  datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      )
  datagen.fit(x_train)


  batch_size = 256
  num_epoch = 60

  model = build_model(with_softmax=True)
  opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
  model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
  #'''
  model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                      steps_per_epoch=x_train.shape[0] // batch_size,epochs=num_epoch,\
                      verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])

  #'''

  model_folder = './models/'
  model.save(model_folder+model_name+'.h5py')
  model.save_weights(model_folder+model_name+'.weights')
  chg_config_model_file(model_folder+model_name+'.h5py')
