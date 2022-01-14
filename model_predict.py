#!--*-- coding:utf-8 --*--
'''
Created on 2020年8月26日

@author: lw
'''


import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization, LeakyReLU,concatenate,Activation
import keras
from keras.models import *
from skimage.io import imread



inputs = Input((512, 512,2))

x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

# x = BatchNormalization()(x)
x1 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal',  name='o1')(x)
 

y = MaxPooling2D(pool_size=(3, 3),strides=(1, 1), padding='same')(x1)
 

bound = keras.layers.subtract([x1, y])
 
# 加激活函数，不加，网络没法后向传播
bound = LeakyReLU(0.2,name='o2')(bound)
 

model = Model(inputs = [inputs], outputs = [x1, bound])



model.load_weights('fourcnnedge.hdf5')


data_path = '/home/lw/data/MultiBandLake/val/satellite/landsat 8/20200819_2/'


img_type = 'tif'
imgs = glob.glob(data_path + "2/*." + img_type)
filenums = len(imgs)

print filenums
 
for i in range(filenums):
    imgpathB2 = data_path + '2/' + str(i)+'.' + img_type
    imgpathB3 = data_path + '3/'+  str(i)+'.' + img_type
    imgpathB4 = data_path + '4/'+  str(i)+'.' + img_type
    imgpathB5 = data_path + '5/'+  str(i)+'.' + img_type
    imgpathB6 = data_path + '6/'+  str(i)+'.' + img_type

#     imgb2 = imread(imgpathB2)
#     imgb3 = imread(imgpathB3)
#     imgb4 = imread(imgpathB4)
    imgb5 = imread(imgpathB5)
    imgb6 = imread(imgpathB6)

#     imgb2 = imgb2 / 255. 
#     imgb3 = imgb3 / 255. 
#     imgb4 = imgb4 / 255.       
    imgb5 = imgb5 / 255. 
    imgb6 = imgb6 / 255. 
#     image = imgb6

#     image = image[np.newaxis, ...,np.newaxis,]
    image = np.stack(( imgb5,imgb6), axis=-1) 
#     image = np.stack((imgb2, imgb3, imgb4, imgb5,imgb6), axis=-1)   
    image = image[np.newaxis, ... ]

    img1,img2 = model.predict(image, batch_size=1, verbose=1)


    img1 = array_to_img(img1[0])
    img2 = array_to_img(img2[0])
    
    
    img1.save("/home/lw/data/MultiBandLake/val/satellite/landsat 8/20200819_2/result1/" + str(i) +'a.'+ img_type )
    img2.save("/home/lw/data/MultiBandLake/val/satellite/landsat 8/20200819_2/result1/" + str(i) +'b.'+ img_type )


 