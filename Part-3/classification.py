#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:22:48 2022

@author: dhaval
"""

import pandas as pd
import cv2
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('classic')

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def clean_image(image_path):
  org_img = cv2.imread(image_path)
  gray_image = cv2.imread(image_path, 0)
  contours, hierachy = cv2.findContours(gray_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  mask = np.zeros(org_img.shape[:2], dtype=org_img.dtype)
  for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      if cv2.contourArea(c) > 25 and cv2.contourArea(c)<6000:
          cv2.drawContours(mask, [c], 0, (255), -1)

  org_img = cv2.bitwise_and(org_img, org_img, mask= mask)
  return org_img

df1 = pd.read_csv(r'Patient_1_Labels.csv')
df2 = pd.read_csv(r'Patient_2_Labels.csv')
df3 = pd.read_csv(r'Patient_3_Labels.csv')
df4 = pd.read_csv(r'Patient_4_Labels.csv')
# df5 = pd.read_csv(r'Patient_5_Labels.csv')

df1.loc[df1['Label'] > 1, 'Label'] = 1
df2.loc[df2['Label'] > 1, 'Label'] = 1
df3.loc[df3['Label'] > 1, 'Label'] = 1
df4.loc[df4['Label'] > 1, 'Label'] = 1
# df5.loc[df5['Label'] > 1, 'Label'] = 1

train = 'train/'
test = 'test/'
val = 'val/'
if os.path.exists('train') and os.path.isdir('train'):
    shutil.rmtree('train')
os.makedirs('train')
if os.path.exists('test') and os.path.isdir('test'):
    shutil.rmtree('test')
os.makedirs('test')
if os.path.exists('val') and os.path.isdir('val'):
    shutil.rmtree('val')
os.makedirs('val')

count = 0

label0 = '0'
label1 = '1'


path = os.path.join(train, label0)
os.mkdir(path)
path = os.path.join(train, label1)
os.mkdir(path)

path = os.path.join(test, label0)
os.mkdir(path)
path = os.path.join(test, label1)
os.mkdir(path)

path = os.path.join(val, label0)
os.mkdir(path)
path = os.path.join(val, label1)
os.mkdir(path)

patients = {'Patient_1/' : df1, 'Patient_2/': df2, 'Patient_3/' : df3}

# train
for key in patients:         
    for index, row in patients[key].iterrows():
        label = int(row['Label'])
        store = '/' + str(count) + '.png'
        if label == 0:
          #print(row['Label'], row['IC'])
          path = 'IC_'+str(row['IC'])+'_thresh.png';
          cleaned_img = clean_image(key + path)
          cv2.imwrite(train + label0 + store, cleaned_img)
        else:
          path = 'IC_'+str(row['IC'])+'_thresh.png';
          cleaned_img = clean_image(key + path)
          cv2.imwrite(train + label1 + store, cleaned_img)
        count = count  +1
 
# test       
# for index, row in df5.iterrows():
#     label = int(row['Label'])
#     store = '/' + str(count) + '.png'
#     if label == 0:
#       #print(row['Label'], row['IC'])
#       path = 'IC_'+str(row['IC'])+'_thresh.png';
#       store = '/' + str(row['IC']) + '.png'
#
#       cleaned_img = clean_image('Patient_5/' + path)
#       cv2.imwrite(test + label0 + store, cleaned_img)
#     else:
#       path = 'IC_'+str(row['IC'])+'_thresh.png';
#       store = '/' + str(row['IC']) + '.png'
#       cleaned_img = clean_image('Patient_5/' + path)
#       cv2.imwrite(test + label1 +  store, cleaned_img)
#     count = count  +1

# val
for index, row in df4.iterrows():
    label = int(row['Label'])
    
    if label == 0:
      #print(row['Label'], row['IC'])
      path = 'IC_'+str(row['IC'])+'_thresh.png';
      store = '/' + str(row['IC']) + '.png'
      
      cleaned_img = clean_image('Patient_4/' + path)
      cv2.imwrite(val + label0 +  store, cleaned_img)
    else:
      path = 'IC_'+str(row['IC'])+'_thresh.png';
      store = '/' + str(row['IC']) + '.png'
      
      cleaned_img = clean_image('Patient_4/' + path)
      cv2.imwrite(val + label1 +  store, cleaned_img)
      

train_path="train"
test_path="test"
val_path="val"

x_train=[]

for folder in os.listdir(train_path):
    if(folder == ".DS_Store"):
        continue
    sub_path=train_path+"/"+folder
    print(folder)

    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(350,350))
        x_train.append(img_arr)

x_test=[]

for folder in os.listdir(test_path):
    if(folder == ".DS_Store"):
        continue
    sub_path=test_path+"/"+folder

    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(350,350))
        x_test.append(img_arr)

x_val=[]

for folder in os.listdir(val_path):
    if(folder == ".DS_Store"):
        continue
    sub_path=val_path+"/"+folder

    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(350,350))
        x_val.append(img_arr)

train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)

train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (350, 350),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (350, 350),
                                            batch_size = 32,
                                            class_mode = 'sparse')
val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (350, 350),
                                            batch_size = 32,
                                            class_mode = 'sparse')

train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes

print(training_set.class_indices)
print(train_y.shape,test_y.shape,val_y.shape)

IMAGE_SIZE =244

vgg = VGG19(input_shape=[350,350, 3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    
x = Flatten()(vgg.output)

prediction = Dense(2, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)


history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=10,
  callbacks=[early_stop],
  batch_size=32,shuffle=True)
model.save('NewClassification.h5')
    
      

