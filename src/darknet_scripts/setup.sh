#!/bin/bash

# cd to darknet root

mkdir models
cd models
wget https://pjreddie.com/media/files/darknet53.conv.74

cd ../
cp custom_setup/split_train_test.py scripts/split_train_test.py
cp custom_setup/voc_label_custom.py scripts/voc_label_custom.py

cp custom_setup/yolov3-tiny-custom.cfg cfg/yolov3-tiny-custom.cfg
cp custom_setup/yolov3-tiny-custom-test.cfg cfg/yolov3-tiny-custom-test.cfg

cp custom_setup/yolov3-custom.cfg cfg/yolov3-custom.cfg
cp custom_setup/yolov3-custom-test.cfg cfg/yolov3-custom-test.cfg
