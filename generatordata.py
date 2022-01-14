# -*- coding: utf-8 -*-  
'''
Created on 2020年7月9日

@author: lw
'''
# -*- coding: utf-8 -*-  
import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np

import glob
from tensorflow.python.distribute.estimator_training import PS
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from operator import ge
 

import skimage.transform as trans

"""
Some lines borrowed from: https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93
"""


def rotate_clk_img_and_msk(img, msk, band):
    angle = np.random.choice((4, 6, 8, 10, 12, 14, 16, 18, 20))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    band_o = trans.rotate(band, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o,band_o

def rotate_cclk_img_and_msk(img, msk, band):
    angle = np.random.choice((-20, -18, -16, -14, -12, -10, -8, -6, -4))
    img_o = trans.rotate(img, angle, resize=False, preserve_range=True, mode='symmetric')
    msk_o = trans.rotate(msk, angle, resize=False, preserve_range=True, mode='symmetric')
    band_o = trans.rotate(band, angle, resize=False, preserve_range=True, mode='symmetric')
    return img_o, msk_o, band_o

def flippingUpDown_img_and_msk(img, msk, band):
    img_o = np.flip(img, axis=0)
    msk_o = np.flip(msk, axis=0)
    band_o = np.flip(band, axis=0)
    return img_o, msk_o, band_o
def flippingLeftRight_img_and_msk(img, msk, band):
    img_o = np.flip(img, axis=1)
    msk_o = np.flip(msk, axis=1)
    band_o = np.flip(band, axis=1)
    return img_o, msk_o, band_o

# 缩放 剪切
def zoom_img_and_msk(img, msk, band):

    zoom_factor = np.random.choice((1.2, 1.5, 1.8, 2, 2.2, 2.5))  # currently doesn't have zoom out!
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    img = trans.resize(img, (zh, zw), preserve_range=True, mode='symmetric')
    msk = trans.resize(msk, (zh, zw), preserve_range=True, mode='symmetric')
    band = trans.resize(band, (zh, zw), preserve_range=True, mode='symmetric')
    
    region = np.random.choice((0, 1, 2, 3, 4))

    # zooming out
    if zoom_factor <= 1:
        outimg = img
        outmsk = msk
        outband = band

    # zooming in
    else:
        # bounding box of the clipped region within the input array
        if region == 0:
            outimg = img[0:h, 0:w]
            outmsk = msk[0:h, 0:w]
            outband = band[0:h, 0:w]
        if region == 1:
            outimg = img[0:h, zw - w:zw]
            outmsk = msk[0:h, zw - w:zw]
            outband = band[0:h, zw - w:zw]
        if region == 2:
            outimg = img[zh - h:zh, 0:w]
            outmsk = msk[zh - h:zh, 0:w]
            outband= band[zh - h:zh, 0:w]
        if region == 3:
            outimg = img[zh - h:zh, zw - w:zw]
            outmsk = msk[zh - h:zh, zw - w:zw]
            outband= band[zh - h:zh, zw - w:zw]
        if region == 4:
            marh = h // 2
            marw = w // 2
            outimg = img[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]
            outmsk = msk[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]
            outband= band[(zh // 2 - marh):(zh // 2 + marh), (zw // 2 - marw):(zw // 2 + marw)]

    # to make sure the output is in the same size of the input
    img_o = trans.resize(outimg, (h, w), preserve_range=True, mode='symmetric')
    msk_o = trans.resize(outmsk, (h, w), preserve_range=True, mode='symmetric')
    band_o = trans.resize(outband, (h, w), preserve_range=True, mode='symmetric')
    return img_o, msk_o, band_o


"""
Some lines borrowed from https://www.kaggle.com/petrosgk/keras-vgg19-0-93028-private-lb
"""

def create_train_data(B2_path, B3_path, B4_path, B5_path, B6_path, label_path, bound_path, img_type = "tif"):


    lbs = glob.glob(label_path + "/*." + img_type)
    filenums = len(lbs)
      
  
    b2datas = []
    b3datas = []
    b4datas = []
    b5datas = []
    b6datas = []
      
    imglabels = []
    imgbounds = []
      
    for i in range(filenums):
        b2path = B2_path +  str(i)+'.' + img_type
        b3path = B3_path +  str(i)+'.' + img_type
        b4path = B4_path +  str(i)+'.' + img_type
        b5path = B5_path +  str(i)+'.' + img_type
        b6path = B6_path +  str(i)+'.' + img_type
          
          
        labelpath = label_path +  str(i)+'.' + img_type
        boundpath = bound_path +  str(i)+'.' + img_type
        b2datas.append(b2path)
        b3datas.append(b3path)
        b4datas.append(b4path)
        b5datas.append(b5path)
        b6datas.append(b6path)
          
        imglabels.append(labelpath)
        imgbounds.append(boundpath)
  
    b2datas = np.array(b2datas)
    b3datas = np.array(b3datas)
    b4datas = np.array(b4datas)
    b5datas = np.array(b5datas)
    b6datas = np.array(b6datas)
          
    imglabels = np.array(imglabels)
    imgbounds = np.array(imgbounds)
      
    return b2datas,b3datas,b4datas,b5datas,b6datas,imglabels,imgbounds


# def a(a,b,c,d,e,f,g,m):
#     r = a+b+c+d+e+f+g+m
#     return r
# 
# c = a(1,2,3,4,5,6,7,8)
# print c





# 读数据
# create_train_data( B2_path = '/home/lw/data/MultiBandLake/images/B2/')
# # print '-----------------------------'
# print imgs_train.shape 
# print imgs_mask_train[0]
# print imgs_bound_train[0]
# 
# 
# b = zip(imgs_train,imgs_mask_train,imgs_bound_train)
# print b[0]






def mybatch_generator_train(zip_list, batch_size, shuffle=True ):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []

        for b2,b3,b4,b5,b6, mask in batch_files:

#             imgb2 = imread(b2)
#             imgb3 = imread(b3)
#             imgb4 = imread(b4)
            imgb5 = imread(b5)
            imgb6 = imread(b6)
            
#             imgb2 = imgb2 / 255. 
#             imgb3 = imgb3 / 255. 
#             imgb4 = imgb4 / 255. 
            imgb5 = imgb5 / 255. 
            imgb6 = imgb6 / 255. 
 
#             image = np.stack((imgb2,imgb3,imgb4,imgb5, imgb6), axis=-1)    
            image = np.stack((imgb5, imgb6), axis=-1)  
#             image = imgb2
#             image = image[..., np.newaxis]
            


            label = load_img(mask, grayscale=True)
            label = img_to_array(label)
            
  

            # 增广数据
#             rndaug = np.random.randint(6, dtype=int)
# 
#             if rndaug == 0:
#                 img, label,bd = flippingUpDown_img_and_msk(img, label,bd)
#             elif rndaug == 1:
#                 img, label,bd = flippingLeftRight_img_and_msk(img, label,bd) 
#             elif rndaug == 2:
#                 img, label,bd = rotate_clk_img_and_msk(img, label,bd) 
#             elif rndaug == 3:
#                 img, label,bd = rotate_cclk_img_and_msk(img, label,bd)
#             elif rndaug == 4:
#                 img, label,bd = zoom_img_and_msk (img, label,bd)

            label /= 255.   # 标签变成0 和 1， 初始标签的值是0和255         
   
            image_list.append(image)
            mask_list.append(label)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, [mask_list])


        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0


# generator=mybatch_generator_train(list(zip(imgs_train,imgs_mask_train,imgs_bound_train)), 2, shuffle=True )
# 
# imgs_train,imgs_mask_train,imgs_bound_train = next(generator)
# print imgs_train.shape
# print imgs_mask_train.shape
# print imgs_bound_train.shape


# img = load_img('/home/lw/data/MultiBandLake/images/B2/0.tif', grayscale=False)
# 
# # img = load_img('/home/lw/eclipse-workspace/Cloud-Net/data/38-Cloud_training/train_blue/blue_patch_1_1_by_1_LC08_L1TP_002053_20160520_20170324_01_T1.TIF', grayscale=False)





# img = load_img('/home/lw/eclipse-workspace/Cloud-Net/data/38-Cloud_training/train_blue/blue_patch_1_1_by_1_LC08_L1TP_002053_20160520_20170324_01_T1.TIF', grayscale=False)
# print img.shape


def mybatch_generator_validation(zip_list, batch_size, shuffle=False):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []

        for b2,b3,b4,b5,b6, mask in batch_files:

#             imgb2 = imread(b2)
#             imgb3 = imread(b3)
#             imgb4 = imread(b4)
            imgb5 = imread(b5)
            imgb6 = imread(b6)
            
#             imgb2 = imgb2 / 255. 
#             imgb3 = imgb3 / 255. 
#             imgb4 = imgb4 / 255. 
            imgb5 = imgb5 / 255. 
            imgb6 = imgb6 / 255. 
 
#             image = np.stack((imgb2,imgb3,imgb4,imgb5, imgb6), axis=-1)   
            image = np.stack((imgb5, imgb6), axis=-1)  
#             image = imgb2
#             image = image[..., np.newaxis]

            
            label = load_img(mask, grayscale=True)
            label = img_to_array(label)
            label /= 255.   # 标签变成0 和 1， 初始标签的值是0和255         
   
            image_list.append(image)
            mask_list.append(label)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        yield (image_list, [mask_list])

        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0

# generator=mybatch_generator_validation(list(zip(imgs_train,imgs_mask_train,imgs_bound_train)), 2, shuffle=True )
#   
# imgs_train,imgs_mask_train,imgs_bound_train = next(generator)
# print imgs_train.shape
# print imgs_mask_train.shape
# print imgs_bound_train.shape
# print type(imgs_bound_train)
# print imgs_bound_train[0]


def mybatch_generatoredge_train(zip_list, batch_size, shuffle=True ):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []
        bound_list = []

        for b2,b3,b4,b5,b6, mask,bound in batch_files:


            imgb2 = imread(b2)
            imgb3 = imread(b3)
            imgb4 = imread(b4)
            imgb5 = imread(b5)
            imgb6 = imread(b6)
            
            imgb2 = imgb2 / 255. 
            imgb3 = imgb3 / 255. 
            imgb4 = imgb4 / 255. 
            imgb5 = imgb5 / 255. 
            imgb6 = imgb6 / 255. 
#             image = imgb6[... , np.newaxis ]


            
            image = np.stack(( imgb5,imgb6), axis=-1)    
#             image = np.stack((imgb2, imgb3, imgb4, imgb5,imgb6), axis=-1)         

            label = load_img(mask, grayscale=True)

            bd = load_img(bound, grayscale=True)
            


            label = img_to_array(label)



            bd = img_to_array(bd)

            print('--------------------------')
            print(bd)

            # 增广数据
#             rndaug = np.random.randint(6, dtype=int)
#  
#             if rndaug == 0:
#                 image, label,bd = flippingUpDown_img_and_msk(image, label,bd)
#             elif rndaug == 1:
#                 image, label,bd = flippingLeftRight_img_and_msk(image, label,bd) 
#             elif rndaug == 2:
#                 image, label,bd = rotate_clk_img_and_msk(image, label,bd) 
#             elif rndaug == 3:
#                 image, label,bd = rotate_cclk_img_and_msk(image, label,bd)
#             elif rndaug == 4:
#                 image, label,bd = zoom_img_and_msk (image, label,bd)

            label /= 255   # 标签变成0 和 1， 初始标签的值是0和255         
            bd /= 255   # 标签变成0 和 1， 初始标签的值是0和255    
   

            image_list.append(image)
            mask_list.append(label)
            bound_list.append(bd)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        bound_list = np.array(bound_list)
        yield (image_list, [mask_list, bound_list])


        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0


# generator=mybatch_generator_train(list(zip(imgs_train,imgs_mask_train,imgs_bound_train)), 2, shuffle=True )
# 
# imgs_train,imgs_mask_train,imgs_bound_train = next(generator)
# print imgs_train.shape
# print imgs_mask_train.shape
# print imgs_bound_train.shape




def mybatch_generatoredge_validation(zip_list, batch_size, shuffle=False):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image_list = []
        mask_list = []
        bound_list = []

        for b2,b3,b4,b5,b6, mask,bound in batch_files:

            imgb2 = imread(b2)
            imgb3 = imread(b3)
            imgb4 = imread(b4)
            imgb5 = imread(b5)
            imgb6 = imread(b6)
            
            imgb2 = imgb2 / 255. 
            imgb3 = imgb3 / 255. 
            imgb4 = imgb4 / 255. 
            imgb5 = imgb5 / 255. 
            imgb6 = imgb6 / 255. 
#             image = imgb6[... , np.newaxis ]

            
            image = np.stack(( imgb5,imgb6), axis=-1)             
#             image = np.stack((imgb2, imgb3, imgb4, imgb5,imgb6), axis=-1)          


            label = load_img(mask, grayscale=True)
            bd = load_img(bound, grayscale=True)


            label = img_to_array(label)
            bd = img_to_array(bd)
            
            label /= 255   # 标签变成0 和 1， 初始标签的值是0和255         
            bd /= 255   # 标签变成0 和 1， 初始标签的值是0和255    
   

            image_list.append(image)
            mask_list.append(label)
            bound_list.append(bd)

        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        bound_list = np.array(bound_list)
        yield (image_list, [mask_list, bound_list])

        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0

# generator=mybatch_generatorbound_validation(list(zip(imgs_train,imgs_mask_train,imgs_bound_train)), 2, shuffle=True )
#   
# imgs_train,imgs_mask_train,imgs_bound_train = next(generator)
# print imgs_train.shape
# print imgs_mask_train.shape
# print imgs_bound_train.shape
# print type(imgs_bound_train)
# print imgs_bound_train[0]

