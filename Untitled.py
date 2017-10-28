
# coding: utf-8

# In[28]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import os
import cv2

import keras
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Reshape, Dropout
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import RMSprop,SGD


# In[11]:


train_files = os.listdir('./data/')
#print files

train_img_path = []
train_label = []

for f in train_files:
    ims  = os.listdir('./data/'+f)
    for im in ims:
        train_img_path.append('./data/'+f+'/'+im)
        train_label.append(f)

        
train_imgs = []

for fx in train_img_path:
    im = cv2.imread(fx,1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(im, (224, 224))
    #print img.shape
    train_imgs.append(img)


# In[12]:


val_files = os.listdir('./val_data/')
print val_files

val_img_path = []
val_label = []

for f in val_files:
    ims  = os.listdir('./val_data/'+f)
    for im in ims:
        val_img_path.append('./val_data/'+f+'/'+im)
        val_label.append(f)

        
val_imgs = []

for fx in val_img_path:
    im = cv2.imread(fx,1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = cv2.resize(im, (224, 224))
    #print img.shape
    val_imgs.append(img)
print len(val_imgs) 


# In[13]:


#for im in val_imgs:
    #plt.figure(im)
 
    #plt.imshow(im)
    #plt.show()


# In[14]:


train_images = np.asarray(train_imgs)
val_images  = np.asarray(val_imgs)
train_labels  = np.asarray(train_label)
val_labels = np.asarray(val_label)
#new_imag = np.insert(all_faces,1,label,axis = 3)
print train_images.shape


# In[15]:


#print train_labels
y_train = np_utils.to_categorical(train_labels)
y_val = np_utils.to_categorical(val_label)
#print y_train
#x_train= train_images.reshape((len(train_images),3,224,224))
#x_val = val_images.reshape((len(val_images),3,224,224))

print train_images.shape


# In[29]:


model = Sequential()
model.add(Convolution2D(128,3,3, input_shape = (224,224,3),activation='relu'))
model.add(Convolution2D(128,3,3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
#model.add(Dropout(0.4))

model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()
rms = RMSprop()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[30]:


hist= model.fit(train_images, y_train,nb_epoch=5,shuffle = True, batch_size=5,
               validation_data=(val_images,y_val))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# In[35]:



