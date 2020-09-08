#!/bin/bash

darknet_location=/home/lloyd/src/darknet
img_location=/home/lloyd/Documents/datasets/lidar/forestry_usyd/labelled_data/detection/hovermap/rasters
anno_location=/home/lloyd/Documents/datasets/lidar/forestry_usyd/labelled_data/detection/hovermap/box_annotations

num_classes=3

# cd $darknet_location/models
# wget https://pjreddie.com/media/files/yolov3-tiny.weights
# wget https://pjreddie.com/media/files/darknet53.conv.74


# max_batches=2000*num_classes*(64/train_bs)
# steps=0.8*max_batches,0.9*max_batches

###############################################################################3

cd $darknet_location

# delete directory if exists
[ -d "tmp_yolo" ] && rm -r tmp_yolo

# setup sub-directories
mkdir tmp_yolo
mkdir tmp_yolo/cfg
mkdir tmp_yolo/train_data
mkdir tmp_yolo/backup

cp cfg/yolov3-tiny-custom.cfg tmp_yolo/cfg/yolov3-tiny-custom.cfg
cp cfg/yolov3-tiny-custom-test.cfg tmp_yolo/cfg/yolov3-tiny-custom-test.cfg

# object.data
cd tmp_yolo/cfg
exec 3<> obj.data
	echo "classes = $num_classes" >&3
	echo "train  = tmp_yolo/train.txt " >&3
	echo "valid  = tmp_yolo/val.txt " >&3
	echo "names = tmp_yolo/cfg/obj.names " >&3 
	echo "backup = tmp_yolo/backup/ " >&3 
exec 3>&-


# obj.names
declare -a arr=("tree" "shrub" "partial")  
j=1
exec 3<> obj.names

	for i in "${arr[@]}"
	do
		echo "$i" >&3   
		let j=j+1
	done
exec 3>&-


# create yolo-type annotation files (from VOC files) and train/val split
cd ../../
python scripts/voc_label_custom.py -d "$anno_location" -c "tree,shrub,partial"   
python scripts/split_train_test.py -d "$img_location" -y "$darknet_location" -s "10"

# copy image and anno files 
anno_yolo_location=`dirname "$anno_location"`/anno_yolo
cp -a $img_location/. $darknet_location/tmp_yolo/train_data
cp -a $anno_yolo_location/. $darknet_location/tmp_yolo/train_data
rm -rf $anno_yolo_location


# train detector
./darknet detector train tmp_yolo/cfg/obj.data tmp_yolo/cfg/yolov3-tiny-custom.cfg models/darknet53.conv.74


# move test config and weights to deploy folder
mkdir tmp_yolo/deploy
cp tmp_yolo/cfg/yolov3-tiny-custom-test.cfg tmp_yolo/deploy/yolov3.cfg
cp tmp_yolo/backup/yolov3-tiny-custom_final.weights tmp_yolo/deploy/yolov3.weights
cp tmp_yolo/cfg/obj.names tmp_yolo/deploy/obj.names


# test detector
#./darknet detector test tmp_yolo/cfg/obj.data tmp_yolo/deploy/yolov3.cfg tmp_yolo/deploy/yolov3.weights raster_0.jpg



