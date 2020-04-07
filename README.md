# dog cat colorization 

The goal of this project is to convert original gray scale dog or cat images to 
colorful vivid images.

And it is an good example supervised learning and tutorial.

## Contents

### dataset 

- cute dog and cat images : https://www.kaggle.com/c/dogs-vs-cats/data

- not use RGB type that is usually used for images but Lab types

- convert from gray images to colorful images

![cat_gray](https://user-images.githubusercontent.com/18729104/77840893-0ab0a200-71c7-11ea-8ef8-53858b3107fc.jpg)
![cat](https://user-images.githubusercontent.com/18729104/77840892-0a180b80-71c7-11ea-8f75-95462a23b57b.jpg)

![dog_gray](https://user-images.githubusercontent.com/18729104/77840896-0b493880-71c7-11ea-82fb-6cacb9b7c29a.jpg)
![dog](https://user-images.githubusercontent.com/18729104/77840895-0b493880-71c7-11ea-857e-40d65783b007.jpg)

### model

- Simple custom U Network

![networks](https://user-images.githubusercontent.com/18729104/78328008-129e8680-75b9-11ea-9c7d-4283f26ab977.jpg)


### loss

- L2 loss (+Softmax)

### train

- epoch : 100

- learning rate : 1e-5

- optimizer : adam

- weight_decay : 1e-5

- batch_size : 16

- loss plot

![loss](https://user-images.githubusercontent.com/18729104/78471203-7d42f400-776a-11ea-8305-967146160497.JPG)

### test

- test images

following images are each gray scale image and predict image and original image

![test01](https://user-images.githubusercontent.com/18729104/78471287-238ef980-776b-11ea-91a8-0f8f2786f599.JPG)
![test02](https://user-images.githubusercontent.com/18729104/78471288-24c02680-776b-11ea-8ea4-53b1b0961e49.JPG)
![test03](https://user-images.githubusercontent.com/18729104/78471291-25f15380-776b-11ea-9b27-cca98309b61a.JPG)
![test04](https://user-images.githubusercontent.com/18729104/78471292-27228080-776b-11ea-8d73-73be8b673c11.JPG)
![test05](https://user-images.githubusercontent.com/18729104/78471293-27bb1700-776b-11ea-8c8a-08117886afdb.JPG)


## Simple guide

### setting 

- visdom

- python >= 3.5

- pytorch >= 1.1.0

- scikit-image

### pre-trained weight 

- you can downloads U-net epoch 99 weight at 
https://drive.google.com/open?id=1dOR2i34aRIy7OrzGrZmgn9Kunef-Q3Lx

### run about your images

- first, you download pre-trained model or train yourself and make pth file in ./saves

- In draw.py, you can change image_dir that you want to colorize and runs draw.py

### future works

- object known, colorizationS