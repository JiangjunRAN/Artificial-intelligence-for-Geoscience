# LaeNet: A Novel Lightweight Multitask CNN  


By Wei Liu, Xingyu Chen, Jiangjun Ran*, Lin Liu, Qiang Wang , Linyang Xin and Gang Li  

Remote Sensing 2021

### Introduction
---

To overcome the aforementioned difficulties, we propose a novel end-to-end lightweight multitask no-downsampling fully convolutional neural Network to segment area and extract edge from remote sensing images simultaneously. We name it the Lightweight area and edge Network (LaeNet) and its architecture is illustrated in Figure 1. Specifically, we firstly pack several no-downsampling and multichannel fully convolutional layers with ReLU activation function as a feature extractor to learn high-level feature map from multiband remote sensing imagery. Then, another no-downsampling and single-channel convolutional layer with Sigmoid activation function is applied to predict lake area and nonarea (land), thereby achieving area segmentation. Based on this, the difference between area segmentation and its spatial gradient is derived as the corresponding predictive edge.

<img src="github.com\grav-cryo\LAE_Net.png" alt="模型结构" style="zoom:50%;" />

### Install and use

------

Please download the zip file containing the code file and install it locally or on a supercomputer (note the corresponding Python 3.6).

First, the user needs to run the toedge.py to generate boundary files for the training data. Then, the user can run the model_cnn_area_edge.py to train the training data, and use the model_predictor.py to predict the test data.



### Setting

------

```
Python package: Numpy, GDAL 2.3, tensorflow-gpu 1.14.0, Keras 2.2.4, pillow, imgaug
Other: CUDA 10.0, cudnn 7.0, conda 4.5.4.
```



### Prepare Data

------

The user needs to prepare the training data set and validation data set required by the model to train the neural network model. The recommended data format is .Tif, and the recommended image size is 512 × 512.



### Citation

------

If codes here are useful for your project, please consider citing our papers:

The scripts for producing our results in paper [Liu et at. 2021](https://www.mdpi.com/2072-4292/13/1/56).

```
@article{
  title={LaeNet: A Novel Lightweight Multitask CNN for Automatically Extracting Lake Area and Shoreline from Remote Sensing Images},
  author={Wei Liu, Xingyu Chen, Jiangjun Ran*, Lin Liu, Qiang Wang, Linyang Xin and Gang Li},
  journal={Remote Sensing},
  volume = {13},
  pages={56},
  year = {2021},
  publisher={MDPI},
  doi = {https://dx.doi.org/10.3390/rs13010056}
}
```
