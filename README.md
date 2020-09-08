# Forest 3D App: inventory from pointcloud data.

![Alt text](media/drawing1.png?raw=true )

This aim of this project is to characterise trees in 3D pointcloud data for forest inventory purposes. The code has been designed for:
- **accuracy**: it uses state-of-the-art machine learning methods and software tools
- **speed**: built off the fast YoloV3 architecture, detection can be done with your CPU in reasonable time, and training models can be done with your GPU
- **flexibility**: the software is made to be easily reconfigurable for different pointcloud types. It has been tested with pointclouds collected with traditional low resolution ALS, high resolution ALS and TLS, and dense photogrammetry.
- **scalability**: developed for a range of scan sizes from single plots to large compartments

All written in python, the app libraries can be found in src/Forest3D. Some example scripts that use the libraries can be found in src/example_detection_scripts. These scripts call pre-trained models to detect and delineate trees in different pointclouds. If the pre-trained models are not good enough and you want to train your own models, this readme explains how you can do that too.


Readme contents:
- [Setup/Installation](#setup/installation)
- [How to use the Forest 3D libraries](#how-to-use-the-forest-3d-libraries)
  - [Reading data](#reading-data)
  - [Ground characterisation and removal](#ground-characterisation-and-removal)
  - [Tree detection and delineation](#tree-detection-and-delineation)
  - [Inventory](#inventory)
- [Pre-trained detection models](#pre-trained-detection-models)
- [How to train your own detection model](#how-to-train-your-own-detection-model)
- [Detection debugging tips](#detection-debugging-tips)
- [Inventory results for several datasets](#inventory-results-for-several-datasets)
- [Recent publications](#recent-publications)


![Alt text](media/drawing6.png?raw=true "Tree delineation using Forest 3D is designed to be flexible to the type of pointcloud and scale from plots to large compartments.")

## Setup/Installation

### Setting Up Forest 3D App

If you are using Ubuntu or Mac OS, clone the repository:
```
git clone ...
```
Recommendation: setup and run within a virtual environment to isolate the python packages from those on your system. To create a blank python3 virtual envrironment:
```
cd ...
virtualenv -p python3 ./venv
```
Python 3.5 definitely works. I have not tested newer versions of python. Activate the virtual environment:
```
source ./venv/bin/activate
``` 
While activated, install the required python packages using pip:
```
pip install -r requirements.txt
```

If you are using Windows, download the repository from Github, and then install the python packages: numpy, scipy, scikit-learn, scikit-image, matplotlib, laspy, plyfile, opencv-python. Check requirements.txt for the correct versions of each.


### Installing Darknet (Optional)

If you want to train your own detection model, you will need to install [darknet](https://pjreddie.com/darknet/). Otherwise, if you already have a model (in the form of a yolov3.weights file and yolov3.cfg file) then you can skip this step. Or if you have another mechanism for training a yolo_v3 detector (e.g. google collab) you can skip this step.

These instructions are for the yolo_v3_tiny model, and rely on Ubuntu or Mac OS operating systems.

Clone the official darknet repo:
```
git clone https://github.com/pjreddie/darknet.git
cd darknet
```

If you have a GPU and CUDA installed (strongly recommended: makes it train significantly faster), change the first line of the makefile to GPU=1. If you intend to only train with a CPU, ignore this.

Then, whether you are using a GPU or CPU, compile darknet with:
```
make
```

Check if it compiled properly with:
```
./darknet
``` 
This should output:
```
usage: ./darknet <function>
```
If you run into issues, look [here](https://pjreddie.com/darknet/install/) or google the error codes output by the console.


## How to use the Forest 3D libraries

There are several Forest 3D libraries you can use by importing them into a python environment. Make sure your python interpretter is set to be the one within the virtual environment. Then in a python script in your own project directory, you can import the libraries by first adding the path of the Forest 3D app source code:
```
import sys
sys.path.insert(0, 'path/to/forest_3d_app/src')

from forest3D import lidar_IO,ground_removal,treeDetector,detection_tools,processLidar,inventory_tools
```

There are libraries for:
- reading (and writing) pointcloud data into (and out of) python
- removing, exporting and importing the ground
- rasterising pointclouds
- detecting and delineating trees
- useful classes for representing pointclouds
- computing inventory

There are also a range of useful tools and utilities for doing a variety of things.


### Reading data

To use the Forest 3D tools such as ground removal and tree detection, your pointcloud data must be read into python as a numpy array. The library forest3D.lidar_IO has several functions for doing this, depending on the format of your pointcoud data (las,ply,asc,csv,off,bin).

```
from forest3D import lidar_IO
```

E.g to read in the xyz coordinates of a las file:
```
xyz_data = lidar_IO.readFromLas('path/to/file.las', fields = ['x','y','z'])
```
If you want to read in the intensity of each point as well:
```
xyz_data = lidar_IO.readFromLas('path/to/file.las', fields = ['x','y','z','Intensity'])
```
And if you are working with a photogrammetry pointcloud that has coloured points:
```
xyz_data = lidar_IO.readFromLas('path/to/file.las', fields = ['x','y','z','red','green','blue'],convert_colours=True)
```
The result of calling these functions is a numpy array variable that holds your pointcloud (i.e xyz_data in the case of the examples). The array has size NxC, where N is the number of points and C is the number of channels. The first three columns of C are x, y and z. Following that are the intensity or coloured channels (if available). In this array format, your data is ready for further processing.

If you have a very dense pointcloud (e.g TLS or photogrammetry data), then it will be slower to read in (and do further processing). Because many of the Forest 3D tools will reduce the resolution anyway, it is better to downsample your pointcloud before reading it into python. To be safe, consider downsampling to a resolution of 5cm. This can be done using third-party pointcloud software such as [cloudcompare](https://www.danielgm.net/cc/).

A note on normalisation of colours and intensity: 
* the intensity is read in as raw values. You will see in some of the scripts that we scale the intensity channel/column to be between 0 and 1. This is done by dividing all values in that column by the upper bound for intensity (50,000)
```
xyz_data[:, 3] /= 50000.0
```
* for the colour channels, by setting convert_colours=True in lidar_IO.readFromLas() the values are read in between 0 and 255. We then scale colour channels in our array to be between 0 and 1:
```
xyz_data[:, 3:] /= 255.0
```

### Ground Characterisation and Removal

The first major processing step is ground characterisation and removal. Using Forest 3D's ground removal library we can remove the ground points from our pointcloud array, save them as a seperate mesh file, and then read them back into python for future use (e.g. ground normalisation for canopy height models or tree heights for inventory).
```
from forest3D import ground_removal
```
If we only have x,y,z data (no intensity or colour), then we can remove the ground points from our array with:
```
xyz_data_gr = ground_removal.removeGround(xyz_data,offset=[0,0,0],thresh=2.0,proc_path='path/to/output_directory')
```

The array that is returned has the ground points removed from it (i.e. xyz_data_gr should have fewer rows than xyz_data). 
A note on the function parameters:
* If a path string is given to *proc_path*, it will output the ground as a ply mesh to that path. If no path is given, nothing will be output. It is useful to establish an output directory where you will store all of the outputs from running Forest 3D functions (such as the ground mesh).
* *thresh* is a single float which gives the distance above the ground mesh under which points are removed. This can be set differently for different tasks, but its primary purpose is to remove the ground and ground vegetation to make other tasks easier. For example, if you are detecting mature pine trees, you should set this to 2 metres (note the units are the same as the pointcloud array units) so that ground vegetation is removed and not falsely detected as a tree. If you want to measure dbh, you will want a seperate ground-removed-pointcloud with a threshold of 1 metre. If you are detecting seedlines you could set this to 0.5 metres. 
* *offset* is just a list of 3 values (x, y and z) to offset the pointcloud with (default [0,0,0]). This should only be used if you notice some array calculations on very large numbers are becoming unstable. Note, if you use a non-zero offset, the returned array will have the offset values. When saving outputs later, you will have the chance to remove the offset so the saved data has the original coordinate frame. 

If you have intensity or colour for each pulse return information, you call the function with the additional *returns* parameter:
```
xyz_data_gr,colour_gr = ground_removal.removeGround(xyz_data,returns=xyz_data[:,3:],offset=[0,0,0],thresh=2.0,proc_path='path/to/output_directory')
```
This returns two arrays: one for xyz data with ground points removed, and one for the colour or return intensity with ground points removed. These arrays have correponsing points along their rows and hence should have the same number of rows.

If you want to load the saved ground mesh back into python, use:
```
ground_pts = ground_removal.load_ground_surface('path/to/output_directory/_ground_surface.ply')
```
This returns a Nx3 array, where N is the number of ground points and the three columns correspond to x, y and z coordinates. The ground points array is useful for other tasks.

### Tree Detection and Delineation

![Alt text](media/drawing3.png?raw=true "Rasters before an after bounding box detections are made.")

After ground points have been removed from the pointcloud, trees can be detected and delineated. This process involves sliding a window over the pointlcoud, pulling out a crop of points, rasterising the pointcloud to a three channel image, using deep learning object detectors to place bounding boxes around trees, and then outputting the set of points within each bounding box. 

The object detection framework used is[Yolov3](https://pjreddie.com/darknet/yolo/). Detection requires a trained yolov3 detection model, which is a folder which includes the following files: 
* yolov3.cfg  - the configuration file for the yolov3 detection network. This file describes the neural network architecture.
* yolov3.weights - the weights that were learnt when the yolov3 network was trained.
* raster_config.json - the raster configuration. The pointcloud must be rasterised in the same way as the rasters that were used to train the yolov3 detector. This config file rasterisation parameters so the process can be replicated.

The detection folder might also contain a file called obj.names. Whilst not needed, this file is useful as it contains the names of the classes the detector is trained to detect. The order of the names, indexed from zero, also gives the classID (e.g. the first class has classID=0, the second has classID=1, etc.).

This step does NOT require a GPU, in fact, it will not use your GPU if you have one.

The library for tree detection is imported with:
```
from forest3D import treeDetector
``` 

Forest3D.treeDetector is a class-based library, that is, following an object-oriented programming paradigm. The main class is called RasterDetector, and a tree detector object of this class can be instantiated using the raster_config.json as follows:
```
import json
detector_addr = 'path/to/detector/folder'
with open(os.path.join(detector_addr, 'raster_config.json')) as json_file:
    config_dict = json.load(json_file)
rasterTreeDetector = treeDetector.RasterDetector(**config_dict )
```
Now that the tree detector object is set up, your pointcloud array can be passed to its sliding_window() method to detect trees:
```
labels = rasterTreeDetector.sliding_window(detector_addr,xyz_data_gr,ground_pts=ground_pts,windowSize = [100,100],stepSize = 80)
```
With method arguments:
* detector_addr - a string with the path to your detector model folder (the same one which has your raster_config.json).
* xyz_data_gr - your x,y,z pointcloud array (i.e. the one you removed the ground points from). Should be Nx3.
* ground_points - array output by ground_removal.load_ground_surface(). Only necessary if using the 'canopy_height' raster layer
* widowSize - size (metres) of window that slides through pointcloud and extracts out a crop to be rasterised.   
* stepSize - the stride of the sliding window (metres). Should be less than the window size.

Trees that fall on the boundary of a sliding window are detected as the 'partial' class. For this reason the step size should be set so that there is an overlap between sliding windows of at least one tree diameter in width (so that every tree appears as a full tree in at least one window). In the above code snippet, the window size is 100 metres and the step size is 80 metres, so the overlap is 20 metres.
The window size is constrained by the raster grid size (number of cells) and resolution (metres/cell). You can check what they are in the raster_config.json. Make sure the sliding window size is less than (grid size x resolution) so that the extracted pointcloud fits inside the raster, leaving a small buffer for safety (the pointcloud crop is not always centered within the raster). E.g. if your grid size is 600x600 cells with a resolution of 0.2 m/cell, than the raster spans 120m x 120m, so a sliding window size of 100m x 100m fits comfortably inside.  

If you want to use raster layers that utilise intensity or colour information, then use the colour_data argument:
```
labels = rasterTreeDetector.sliding_window(detector_addr,xyz_data_gr,colour_data=intensity_gr,ground_pts=ground_pts,windowSize=[100,100],stepSize=80)
```
The array intensity_gr should be NxC where C is the number of colour channels (1 for just intensity, 3 for r,g,b, 2 for principle components, etc.)

By default the method returns a pointcloud array of size Nx1, whose values delineate each tree by indicating which point belongs to which tree (i.e. each point has a tree ID). Zero is the background ID for any points that don't belong to a tree.

The method can also return the bounding boxes instead of labelled points if the method argument returnBoxes=True. This returns a Mx4 numpy array, where M is the number of detections. The bounding boxes are of the form [ymin xmin ymax xmax] (in the units and coordinate space of the pointcloud). If you want to label the pointcloud with these bounding boxes, use:
```
labels = detection_tools.label_pcd_from_bbox(xyz_data, boxes, xyxy=True)
```

Some other arguments of RasterDetector.sliding_window() that can be tuned are:
* classID - integer that specifies the class of interest. Default is 0. Check obj.data to see the class ID's of different classes
* confidence_thresh - the threshold on bounding box confidence. Default 0.5. Setting it closer to 0 finds more bounding boxes (possibly of lower quality), setting closer to 1 finds fewer.
* overlap_thresh - a threshold for removing overlapping boxes. This is an important parameter to tune, based on the distance between trees. Default 5 for mature Tumut pines. Can set to smaller for other datasets (e.g. for seedlings use overlap_thresh=1, Rotorua hovermap use overlap_thresh=2)


You can output the delineated pointcloud as a ply file for visualisation using:
```
lidar_IO.writePly_labelled(os.path.join(output_dir,'detection.ply'),xyz_data_gr,labels,offset=[0,0,0,0])
```
xyz_data_gr should have the same number of rows as labels. output_dir is the path to your output directory. Offset is optional here (default [0,0,0,0]). If you used an offset before to avoid unstable calculations, then put the offset here to revert the pointcloud to the correct coordinate space.


### Inventory

The inventory toolbox uses the delineated pointcloud to output high-level inventory information, such as the coordinates and height of each tree. To use:
```
from forest3D import inventory_tools
```

Using the pointcloud with matching delineation labels output by the detector function, you can get the x,y,z coordinates of each tree top as an array:
```
tree_tops = inventory_tools.get_tree_tops(xyz_data_gr,labels)
```
Each tree top also has a tree ID (signifying which tree in labels it belongs to). This is the fourth column of tree_tops.

Once you have the tree top coordinates, you can get the height of each tree using the ground:
```
heights = inventory_tools.get_tree_heights(tree_tops[:,:3],ground_pts)
```
ground_pts is the array of ground points loaded with ground_removal.load_ground_surface().

The tree tops, id's and heights can then all be saved as a csv:
```
inventory = np.hstack((tree_tops,heights[:,np.newaxis]))
utilities.write_csv(os.path.join(output_dir,'inventory.csv'),inventory,header='x,y,z,id,height')
```
It is possible to load this csv into a 3D viewer like [CloudCompare](https://www.danielgm.net/cc/), where it can be overlaid with the original pointcloud.

## Pre-trained Detection Models

Several detection models have been trained on forest pointcloud data and are available for use. The table describes each model. The appropriate model for your dataset should be the model trained on data most similar to your own dataset. Make sure you have return intensity or colour information for each point if the model requires it. 

| Model name        | Dataset used for training| Location | Species/Age  | Sensor | Description  | 
| ------------- |:-------------:| -----:| -----:| -----:| -----:|
| tumut1      | V1_Scanner1_161011_220153_crop001.las  | Tumut NSW |Mature Pine |  VUX-1 ALS (helicopter) | High res pointcloud. Only trained on x,y,z data (no return intensity).
| tumut2      | V1_Scanner1_161011_220153_crop001.las  | Tumut NSW |Mature Pine |  VUX-1 ALS (helicopter) | High res pointcloud. Trained on x,y,z and return intensity data.
| hvp         | saltwater_31B_1_2.las  | HVP VIC |Mature Pine |  ? ALS ? | Low res ALS. Only trained on x,y,z data (no return intensity).
| utas | transect_large_vis_dense_down5cm.las | Tasmania | Pine Seedling |  Drone Photogrammetry | Dense photogrammetric pointcloud. Trained on x,y,z with 2 principle components of red,green,blue for each point. 
| hovermap | Interpine_02_Output_laz1_2_down5cm.las  | Rotorua, NZL |Mature Pine | Hovermap backpack | High res under-canopy data similar to TLS. Trained on x,y,z and return intensity data.

If '_down5cm' is appended to the filename as in xxx_down5cm.las, it means the original dataset xxx.las was downsampled to 5cm resolution.

Acknowledgements to Interpine, UTAS, Forect Corp and the other project partners responsible for collecting the data used to train the models.  


## How to train your own detection model

To do this, you will need some have some raster data with labels and also have installed Darkent (see [Installing Darknet](#Installing Darknet) above).

### Generating training rasters

The script **gen_training_rasters.py** can be used to generate raster jpg's from pointcloud data. There are a few different ways to run this script depending on the kind of data you have:
* you have several pointcloud crops (not limited to circles) you wish to convert to rasters
* you have a single large pointcloud and have a csv file with the locations of circle centers and radii
* you have a single large pointcloud and have several csv's, one for each circle, that contain the cropping circle center and radius information


Note, this process removes the ground from the pointclouds, as well as points above the ground up to a height threshold. This is to remove most of the ground vegetation from the rasters, making it easier to detect trees.

Many ways to do (a few shown in gen_rasters scripts). Just have to remember: output config, and make sure pointcloud fits within raster (gridsize and res)

Can generate rasters with annotations so assist with bounding box labels. E.g. if you have text files with differential gps measurements.

### Raster Configuration

```
rasterTreeDetector = treeDetector.RasterDetector( raster_layers, support_window=[1,1,1],normalisation='rescale',doHisteq=[True,True,True],res=0.2,gridSize=[600, 600, 1000] )
```
The object created has a specific way of rasterising pointcloud data to a three channel colour image, which is given by its construction parameters:
* raster_layers - a list of between one and three strings, where each string is a per channel rasterisation approach E.g. ['vertical_density', 'canopy_height', 'mean_colour1']. It is sufficient to only put one name instead of three if the 'cmap_jet' normalisation approach is used as it only requires one channel. If another normalisation approach is used, then non-specified raster channels will be information-less rasters of uniform values (i.e. 0.5). Options are: 'vertical_density', 'max_height', 'canopy_height', 'mean_colour1', 'mean_colour2', 'mean_colour3', 'max_colour1', 'max_colour2', 'max_colour3'
* support_window - a list of ints which specify the size of neighbourhoods used to rasterise points. This is useful for very sparse pointclouds such as low res ALS.
* normalisation - a single string specifying the normalisation approach. Options are 'rescale', 'rescale+histeq' and 'cmap_jet'.
* doHisteq - list of booleans relating to normalisation. Only applicable if using 'rescale+histeq' normalisation.
* res - float specifying the resolution of each raster grid cell (in metres)
* gridSize - list of ints specifying the number of cells in each axis of the grid (x,y,z).

res determines speed, size of objects and sparsity.

Smaller res increases object size (easier for detector), but descreases speed and increases sparsity (can be counteracted with larger support windows)

max_colour1 refers to first colour input, max_colour2 refers to second,etc. 

description of each raster layer

### Labelling rasters with bounding boxes

Once you have some rasters, you can label them with bounding boxes. We suggest using a tool called [labelImg](https://github.com/tzutalin/labelImg), but you can use any tool you want as long as it can output the labels as XML files in PASCAL VOC format.

Once you have installed labelImg, point it to your directory of rasters and a directory where you wish to save you bounding box labels. Then start drawing boxes around ALL objects of interest in each raster. The detection script is designed for three classes:
* **tree** 
* **shrub** - ground vegetation (there shouldn't be too much of this because we remove alot of it).
* **partial** - these are trees that get cut in half because they lie on the boundary of where the pointcloud is cropped. It is important to detect these as a seperate class to trees so that they are ignored and the full detetion (which we will get if our sliding window overlaps a bit) is used instead.

If you are using the detection script *as-is*, it is very important to use these class names when labelling (don't make a spelling mistakes). If you want to speed up your labelling, check out the hotkeys at [labelImg](https://github.com/tzutalin/labelImg). Also, you might find it handy to look at the data in 3D as well to get a more complete perspective of the raster (just open the pointcloud with your 3D viewer) .

Make sure the bounding box label files (the .xml files) have the same name as their corresponsing image (.jpg) file. This should happen by default if you use the labelImg tool.

*Talk about how gen_rasters can output rasters with detections highlighted (using other annotation information)

### Training a yolov3 model with Darknet

Make sure you have Darknet installed. If you are training your first model, from forest_3d_app/src/darknet_scripts copy setup.sh and the custom_setup/ folder to your Darknet root. Then run:
```
./setup.sh
```
This copies two python scripts and two config files to where they need to be in the Darknet root. It will also download some weights pre-trained on Imagenet to initialise the convolutional layers for yolov3. The config files specify the architecture and training configuration for training and prediction. The python scripts are for converting the bounding box xml labels from PASCAL VOC format to YOLO format, and splitting data into test and training data. 

Edit header. 
Change epochs.

Different class num

train until 0.06

Instructions for using custom classes: two places in auto_setup_yolo.sh where you need to change the class names. Order here is important (determines class numbers)
The file obj.names gives the classes and their class numbers (based on their index)



## Detection Debugging Tips

Trying out your detector on a full pointcloud can take a couple of minutes, which is annoying if you have to tune parameters or even check if it works. Break up the process with some intermediate testing steps. 

First you check for errors in the detector. This test isolates the detector from pointcloud processing errors. Use darknet to try out your detector on a raster image (the kind of one made by a gen_training_rasters python script) using the bash command:
```
./darknet detector test tmp_yolo/deploy/obj.data tmp_yolo/deploy/yolov3.cfg tmp_yolo/deploy/yolov3.weights raster_0.jpg
```
Check to see if the correct classes are being picked up, the bounding boxes are fitting the objects well, and there aren't too many false positive detections (commission errors) or false negative detections (omission errors). You might not have trained your detector for long enough, or maybe there was an error in the training process. This is the easiest point at which to catch most problems with the actual detector.

Once you know the detector works, the next test looks for errors in the pointcloud processing. Crop out a small section of the pointcloud using a tool like [CloudCompare](https://www.danielgm.net/cc/) and run that las file through your code. Look at the delineated pointcloud and see if it matches what you saw with the raster image test. If not, you might need to look at parameters like overlap_thresh in the sliding_window() method. There could also be a problem with your python script (e.g. data is being read in wrong, not rasterising properly). If that is the case you need to step through your code and check the outputs at each stage to make sure they are what you expect.


## Inventory Results for Several Datasets

The table below contains plot-level inventory outputs for several datasets, as computed by the Forest3D tools and models from this project. The name of the script in /example_detection_scripts is given so you can replicate the results.

| Dataset   | Filename | Model used | Script | No. trees detected  | Typical height range (m) | Mean height (m)  |  Std height (m) |
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:| -----:|
| Tumut (high res ALS) | V1_Scanner1_161011_220153_crop001.las  | tumut1 | tumut_detect1.py |  523 | 22-49 | 39.68 | 6.31 
| Tumut FCNSW (high res ALS) | FCNSW_FertTrialAOI_VUX1_conv.las  | tumut1 | tumut_detect1.py |  1626 | 19-33 | 25.99 | 2.55 
| HVP (low res ALS) | saltwater_31B_1_2.las  | hvp | hvp_detect.py |  8693 | 15-31 | 23.95 | 3.50 
| UTAS seedlings | transect_large_vis_dense_down5cm.las  | utas | utas_detect.py |  3618 | 0.39-3.69 | 2.18 | 1.47 
| Rotorua Hovermap | Interpine_02_Output_laz1_2_down5cm.las  | hovermap | hovermap_detect.py |  811 | - | - | - 
| Rotorua Hovermap (high res crop) | Interpine_02_Output_laz1_2_down5cm.las  | hovermap | hovermap_detect.py |  45 | 32-42 | 38.93 | 2.06 

Tree-level inventory outputs can be found in the /media folder as *.csv files. These can be opened with excel, or visualised with the pointcloud using [CloudCompare](https://www.danielgm.net/cc/). 

![Alt text](media/drawing5.png?raw=true "Height distribution of trees mapped spatially for several datasets.")



## Related publications

Some links to open-access publications that use these methodologies for processing 3D forestry data:

- Windrim and Bryson. [Detection, Segmentation, and Model Fitting of Individual Tree Stems from Airborne Laser Scanning of Forests Using Deep Learning](https://www.mdpi.com/2072-4292/12/9/1469). Remote Sensing 12.9 (2020).
- Windrim and Bryson. [Forest Tree Detection and Segmentation using High Resolution Airborne LiDAR](https://arxiv.org/abs/1810.12536). 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2019).

Links to other open-access forestry publications from our group:
- Windrim, Bryson, McLean, Randle, Stone. [Automated Mapping of Woody Debris over Harvested Forest Plantations Using UAVs, High-Resolution Imagery, and Machine Learning](https://www.mdpi.com/2072-4292/11/6/733). Remote Sensing 11.6 (2019).
- Windrim, Carnegie, Webster, Bryson. [Tree Detection and Health Monitoring in Multispectral Aerial Imagery and Photogrammetric Pointclouds Using Machine Learning](https://ieeexplore.ieee.org/abstract/document/9102401). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2020).

