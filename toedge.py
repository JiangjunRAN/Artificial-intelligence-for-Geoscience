# -*- coding: utf-8 -*-
'''
Created on 2020年7月19日
根据 分块标签 批量生成 边界标签

@author: lw 刘伟 
'''
import cv2


# 分块标签所在目录
path1 = 'D:/0DeepLearning/1MultiBand_train_data/5label_data_region/3resubset_label/19_test/'

# 生成边界标签所在目录
path2 = 'D:/0DeepLearning/1MultiBand_train_data/5label_data_region/3resubset_label/edge/'

fnums = 121  # 文件个数

for i in range(fnums):
    fname = str(i) +'.tif'
    oldname = path1 + fname
     
    newname = path2 + fname

    img = cv2.imread(oldname, 0)
    img = cv2.GaussianBlur(img,(3,3),0)
    canny = cv2.Canny(img, 0, 150)
    cv2.imwrite(newname, canny)





