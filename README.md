# dog cat colorization 

The goal of this project is to convert original gray scale dog or cat images to 
colorful vivid images.

And it is an good example supervised learning and tutorial.

## contents

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

![colorization loss](https://user-images.githubusercontent.com/18729104/77840958-bd810000-71c7-11ea-8420-723d3473b457.JPG)

### test


## simple guide

### Setting 

- visdom

- python 3.5 >=

- pytorch 1.1.0

- scikit-image

### 
