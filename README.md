# Google FaceNet with Keras Age & Gender Classification

*April 2018*

The demo combines the face detection from Google FaceNet and the age&gender classification from Keras. 
You can refer the following links for further details:
* [Google FaceNet](https://github.com/davidsandberg/facenet)
* [Age & Gender classification using Keras](https://github.com/yu4u/age-gender-estimation)

Demo video: [link](https://www.youtube.com/watch?v=8fZGE9BlwQw)

### Environment
* Ubuntu 16.04
* Tensorflow 1.5.0
* Keras 2.0.8
* OpenCV 3.2.0
* Python 3.5.2

> Kindly note that it really matters to apply compatible version of each library to speed up the installations. I tried several combinations and came out with the above library versions.

### Hardware
* CPU: *(TBD)*
* GPU: *(TBD)*

### How to run the demo
1. Download the FaceNet library from its [offical repository](https://github.com/davidsandberg/facenet) and put it under the root directory.
2. Run *main.py* 

> Some other issues are decribed in *main.py*

### Issues
| Issue        | Solution           | 
| ------------- |:-------------:| 
| Error message: ValueError: Tensor Tensor("dense_1/Softmax:0", shape=(?, 2), dtype=float32) is not an element of this graph.     | Add *age_gender_model._make_predict_function()* (please refer to line 38 in main.py) | 

> Please feel free to share the issues you meet in the comment below so that we can establish a more comprehensive Q&A list. 
