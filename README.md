# YOLO_tensorflow

(Version 0.3, Last updated :2018.04.02)

### 1.Introduction

This is tensorflow implementation of the YOLO:Real-Time Object Detection

Original code(C implementation) & paper : http://pjreddie.com/darknet/yolo/

My code is for algorithm studying:
Using low level tensorflow code, without tf.contrib.slim, tflearn.Trainer() etc

Two main reference codes for my implementation:  
https://github.com/gliese581gg/YOLO_tensorflow/YOLO_small_tf.py  
https://github.com/hizhangp/yolo_tensorflow


### 2.Install   
(1) Download code

(2) Download YOLO weight file from

YOLO_small : https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing

(3) Put the 'YOLO_small.ckpt' in the 'data\weights' folder of downloaded code

(4) Download Pascal VOC dataset, and create correct directories
	```Shell
	$ ./download_data.sh
	```
(5) Modify configuration in `yolo/config.py`

### 3.Usage
  Training
	```Shell
	$ python train.py
	```
  Test
	```Shell
	$ python test.py
	```
### 4.Requirements

- Tensorflow
- Opencv2

### 5.Copyright

According to the LICENSE file of the original code, 
- Me and original author hold no liability for any damages
- Do not use this on commercial!






