# colorization 
이 프로젝트의 목표는 gray_scale 의 이미지를 색을 칠하는 것 입니다.
- - -
method : deep learning
- - -
model : Alex-net 을 이용한 Auto-Encoder : 모델 변경

1. using auto encoder
![ae](https://user-images.githubusercontent.com/18729104/45923771-f92f6f80-bf29-11e8-9142-7320ed0c8506.png)
***
2. using eccv 2016 colorful image colorization model
![model](https://user-images.githubusercontent.com/18729104/46243967-5f2d6280-c414-11e8-939e-82abad264b24.jpg)

- - -
dataset : 귀여운 강아지와 고양이의 이미지 link : https://www.kaggle.com/tongpython/cat-and-dog

[변경] : 고양이의 데이터만 가지고 실시 (6228 개의 이미지 pair)
***
loss : MSE + regularization (예정)
***
![g_cat 1](https://user-images.githubusercontent.com/18729104/45923723-88d41e80-bf28-11e8-8944-4450ebb0be21.jpg)
![cat 1](https://user-images.githubusercontent.com/18729104/45923706-12cfb780-bf28-11e8-8642-139ba7b07038.jpg)
- - -
2018.09.21 - 프로젝트구성
***
2018.09.23 - 데이터베이스 구성
* * *
2018.09.26 - Data Set 부분 구성
***
2018.09.27 - Model 부분 구성(1) encoder
***
2018.09.28 - Model 부분 구성(2) decoder -- pooling 을 없애니까 잘 됨
***
2018.09.29 - Model 을 변경하여 평가 ( 중간에 concatenate 해봄) Richard Zhang, colorful image colorization, eccv 2016
***
result 01 : 2018.09.28
![result1](https://user-images.githubusercontent.com/18729104/46208199-ee7b3d00-c364-11e8-8d40-7d2e3261aede.JPG)
***
result 02 : 2018.09.29 ( 10 epoch )
![result2](https://user-images.githubusercontent.com/18729104/46243968-62c0e980-c414-11e8-9d89-1513564e960a.png)
***
result 03 : 2018.09.29 ( 30 epoch )
