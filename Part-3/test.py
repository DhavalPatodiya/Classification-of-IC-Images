#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:47:17 2022

@author: dhaval
"""

import pandas as pd

import cv2
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('classic')

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def clean_image(image_path):
    org_img = cv2.imread(image_path)
    gray_image = cv2.imread(image_path, 0)
    contours, hierachy = cv2.findContours(gray_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(org_img.shape[:2], dtype=org_img.dtype)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 25 and cv2.contourArea(c) < 6000:
            cv2.drawContours(mask, [c], 0, (255), -1)

    org_img = cv2.bitwise_and(org_img, org_img, mask=mask)
    return org_img


df5 = pd.read_csv(r'test_Labels.csv')
df5.loc[df5['Label'] > 1, 'Label'] = 1

test = 'test/'

if os.path.exists('test') and os.path.isdir('test'):
    print("gg")
    shutil.rmtree('test')
os.makedirs('test')

label0 = '0'
label1 = '1'

path = os.path.join(test, label0)
os.mkdir(path)
path = os.path.join(test, label1)
os.mkdir(path)

src = 'testPatient/'

# test
for index, row in df5.iterrows():
    label = int(row['Label'])

    if label == 0:
        # print(row['Label'], row['IC'])
        path = 'IC_' + str(row['IC']) + '_thresh.png';
        store = '/' + str(row['IC']) + '.png'

        cleaned_img = clean_image(src + path)
        cv2.imwrite(test + label0 + store, cleaned_img)
    else:
        path = 'IC_' + str(row['IC']) + '_thresh.png';
        store = '/' + str(row['IC']) + '.png'
        cleaned_img = clean_image(src + path)
        cv2.imwrite(test + label1 + store, cleaned_img)

test_path = "test"

x_test = []

for folder in os.listdir(test_path):
    if (folder == ".DS_Store"):
        continue
    sub_path = test_path + "/" + folder

    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (350, 350))
        x_test.append(img_arr)

test_x = np.array(x_test)
test_x = test_x / 255.0
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(350, 350),
                                            batch_size=32,
                                            class_mode='sparse')

test_y = test_set.classes

from keras.models import load_model

model = load_model('NewClassification.h5')

mythreshold = 0.808
from sklearn.metrics import confusion_matrix, classification_report

metrics = []
y_pred = model.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)
cm = confusion_matrix(y_pred,test_y )
print(cm)
#get classification report
print(classification_report(y_pred,test_y))

tn = cm[0][0]
fn = cm[1][0]
tp = cm[1][1]
fp = cm[0][1]

acc = (tp + tn) / (tp + fp + fn + tn) * 100
acc = str(acc) + "%"
print(acc)
metrics.append(["Accuracy", (acc)])

prec = tp / (tp + fp) * 100
prec = str(prec) + "%"
print(prec)
metrics.append(["Precision", (prec)])

sen = tp / (tp + fn) * 100
sen = str(sen) + "%"
print(sen)
metrics.append(["Sensitivity", (sen)])

spec = tn / (tn + fp) * 100
spec = str(spec) + "%"
print(spec)
metrics.append(["Specificity", (spec)])

df = pd.DataFrame(metrics, columns=['Metric', 'Value'])
df.to_csv("Metrics.csv", index=False)

result = []

for folder in os.listdir(test_path):
    if (folder == ".DS_Store"):
        continue
    sub_path = test_path + "/" + folder

    for img in os.listdir(sub_path):
        name = os.path.basename(img)
        name = os.path.splitext(name)[0]
        print(name)
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (350, 350))
        x_test = (img_arr)
        test_x = np.array(x_test)
        test_x = test_x / 255.0
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_set = test_datagen.flow_from_directory(test_path,
                                                    target_size=(350, 350),
                                                    batch_size=32,
                                                    class_mode='sparse')

        test_y = test_set.classes
        img = test_x
        input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
        prediction = np.argmax(model.predict(input_img)[0])
        result.append([name, prediction])

df = pd.DataFrame(result, columns=['IC_Number', 'Label'])
df.to_csv("Results.csv", index=False)


