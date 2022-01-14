#!--*-- coding:utf-8 --*--
'''
Created on 2020年8月26日

@author: lw
'''

from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization, LeakyReLU,concatenate,Activation
from keras.optimizers import *
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from generatordata import create_train_data, mybatch_generatoredge_train, mybatch_generatoredge_validation
from metric import accuracy, precision, recall, f1score, miou
from skimage.io import imread
import numpy as np




# import syft as log
class ADAMLearningRateTracker(keras.callbacks.Callback):
    """It prints out the last used learning rate after each epoch (useful for resuming a training)
    original code: https://github.com/keras-team/keras/issues/7874#issuecomment-329347949
    """

    def __init__(self, end_lr):
        super(ADAMLearningRateTracker, self).__init__()
        self.end_lr = end_lr

    def on_epoch_end(self, epoch, logs={}):  # works only when decay in optimizer is zero
        optimizer = self.model.optimizer
        print('\n***The last Basic Learning rate in this epoch is:', K.eval(optimizer.lr), '***\n')
        # stops the training if the basic lr is less than or equal to end_learning_rate
        if K.eval(optimizer.lr) <= self.end_lr:
            print("training is finished")
            self.model.stop_training = True
###############  GPU 设置 #####################################################

 

num_of_channels = 3
num_of_classes = 1
starting_learning_rate = 1e-4
end_learning_rate = 1e-8
max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
patience = 2
decay_factor = 0.7
batch_sz = 4
######## 网络###################################

inputs = Input((512, 512,2)) # 3通道
# 网络结构定义

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



# cross_entropy_balanced  mse mae

model.compile(optimizer=Adam(lr=1e-2), loss={ 'o1':'binary_crossentropy',  'o2': 'mae'}, 
              loss_weights={ 'o1': 1.0,  'o2': 1.0  },
              metrics={'o1':['accuracy', accuracy, precision, recall, f1score, miou], 'o2':['accuracy']})

model.summary()




# 读数据
b2datas,b3datas,b4datas,b5datas,b6datas,imgs_mask_train, imgs_bound_train = create_train_data('D:/0DeepLearning/1MultiBand_train_data/4train_data_rigion/3select_rectangle/1train_data/B2/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/4train_data_rigion/3select_rectangle/1train_data/B3/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/4train_data_rigion/3select_rectangle/1train_data/B4/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/4train_data_rigion/3select_rectangle/1train_data/B5/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/4train_data_rigion/3select_rectangle/1train_data/B6/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/5label_data_region/3resubset_label/19_test/',
                                                                                              'D:/0DeepLearning/1MultiBand_train_data/5label_data_region/3resubset_label/edge/',
                                                                                              img_type = "tif")




# 训练集 656 100 543 1096  
train_b2 = b2datas[0:100]
train_b3 = b3datas[0:100]
train_b4 = b4datas[0:100]
train_b5 = b5datas[0:100]
train_b6 = b6datas[0:100]


train_msk = imgs_mask_train[0:100]
train_bound = imgs_bound_train[0:100]

# # 打乱
num_examples=len(train_b2)
perm = np.arange(num_examples)
np.random.shuffle(perm)
 
train_b2 = train_b2[perm]
train_b3 = train_b3[perm]
train_b4 = train_b4[perm]
train_b5 = train_b5[perm]
train_b6 = train_b6[perm]

train_msk = train_msk[perm]
train_bound = train_bound[perm]

# 验证集
val_b2 = b2datas[100:]
val_b3 = b3datas[100:]
val_b4 = b4datas[100:]
val_b5 = b5datas[100:]
val_b6 = b6datas[100:]

val_msk = imgs_mask_train[100:]     
val_bound = imgs_bound_train[100:] 


print("loading data done")


# 学习率衰减策略
lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
 
# CSVLogger:将epoch的训练结果保存在csv中
 
csv_logger = CSVLogger('fourcnn_log_1.log')


print("got model_unet")
model_checkpoint = ModelCheckpoint('fourcnnedge.hdf5', monitor='val_o1_acc',verbose=1, save_best_only=True)
print('Fitting model...')


model.fit_generator(
    generator=mybatch_generatoredge_train(list(zip(train_b2,train_b3,train_b4,train_b5,train_b6, train_msk,train_bound)),  batch_size = batch_sz, shuffle=True ),
    steps_per_epoch=np.ceil(len(train_b2) / batch_sz), epochs=max_num_epochs, verbose=1,
    validation_data=mybatch_generatoredge_validation(list(zip(val_b2,val_b3,val_b4,val_b5,val_b6, val_msk,val_bound)) , batch_size = batch_sz , shuffle=False),
    validation_steps=np.ceil(len(val_b2) / batch_sz),
    callbacks=[model_checkpoint,lr_reducer,ADAMLearningRateTracker(end_learning_rate), csv_logger])


