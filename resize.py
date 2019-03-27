#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:46:00 2019

@author: sergiorobledo
"""


'''             '''
''' Resize data '''
'''             '''

data_path = '/Users/sergiorobledo/Desktop/Dermatologist AI/data'

import glob
train_mf = glob.glob(data_path + '/train/melanoma/*.jpg')
train_nf = glob.glob(data_path + '/train/nevus/*.jpg')
train_sf = glob.glob(data_path + '/train/seborrheic_keratosis/*.jpg')

valid_mf = glob.glob(data_path + '/valid/melanoma/*.jpg')
valid_nf = glob.glob(data_path + '/valid/nevus/*.jpg')
valid_sf = glob.glob(data_path + '/valid/seborrheic_keratosis/*.jpg')

test_mf = glob.glob(data_path + '/test/melanoma/*.jpg')
test_nf = glob.glob(data_path + '/test/nevus/*.jpg')
test_sf = glob.glob(data_path + '/test/seborrheic_keratosis/*.jpg')

from PIL import Image
import os

def resize(filelist, folder_name):
    for item in range(len(filelist)):
        root, ext = os.path.splitext(filelist[item])
        im = Image.open(filelist[item])
        imResize = im.resize((256,256))
        imResize.save(data_path + folder_name + '/resized/' + root[-12:] + '.jpg', 'JPEG')
        
resize(train_mf, '/train/melanoma')
resize(train_nf, '/train/nevus')
resize(train_sf, '/train/seborrheic_keratosis')

resize(valid_mf, '/valid/melanoma')
resize(valid_nf, '/valid/nevus')
resize(valid_sf, '/valid/seborrheic_keratosis')

resize(test_mf, '/test/melanoma')
resize(test_nf, '/test/nevus')
resize(test_sf, '/test/seborrheic_keratosis')

