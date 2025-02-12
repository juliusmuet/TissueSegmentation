# TissueSegmentation

This repository contains code for the development, training, and evaluation of machine learning models (CNNs & Transformers) designed for the segmentation of tissue samples using infrared-microscopy data. The project is conducted as part of the "Big Data in Bioinformatics" course at Ruhr-University Bochum. The goal of this specific project was to predict the type of tissue using the 427-dimensional data stored at each pixel of the image.

This framework can be used for any kind of multi-class classification using deep learning.
It offers a high flexibility regarding the amount of classes, the amount of arrays in which the data is stored and the dimensionality of the data.


## Directory Structure

There is a directory structure for the data which is illustrated below.
```
data/
│-- train_data/
│   │-- class_1/
│   │   │-- sample1.npy
│   │   │-- sample2.npy
│   │-- class_2/
│   │   │-- sample1.npy
│   │   │-- sample2.npy
│   │-- ...
│
│-- test_data/
│   │-- class_1/
│   │   │-- sample1.npy
│   │   │-- sample2.npy
│   │-- class_2/
│   │   │-- sample1.npy
│   │   │-- sample2.npy
│   │-- ...
│
│-- statistics/ (stores outputs)
│   │-- classification_report.txt
│   │-- confusion_matrix.png
│
│-- images/
│   │-- image1.npy
│   │-- image2.npy
│   │-- ...
│
│-- images_results/ (stores outputs)
│   │-- segmented_image1.png
│   │-- segmented_image2.png
│   │-- ...
```
