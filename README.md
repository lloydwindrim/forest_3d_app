# Forest 3D App: inventory from pointcloud data.

![Alt text](media/drawing1.png?raw=true)

The aim of this project is to characterise trees in 3D pointcloud data for forest inventory purposes. The code has been designed for:

- **accuracy**: it uses state-of-the-art machine learning methods and software tools
- **speed**: built off the fast YoloV3 architecture, detection can be done with your CPU in reasonable time, and training models can be done with your GPU
- **flexibility**: the software is made to be easily reconfigurable for different pointcloud types. It has been tested with pointclouds collected with traditional low resolution ALS, high resolution ALS and TLS, and dense photogrammetry.
- **scalability**: developed for a range of scan sizes from single plots to large compartments

All written in python, the app libraries can be found in src/Forest3D. Some example scripts that use the libraries can be found in src/example_detection_scripts. These scripts call pre-trained models to detect and delineate trees in different pointclouds. If the pre-trained models are not good enough and you want to train your own models, this readme explains how you can do that too.

Authors: Dr. Lloyd Windrim and Dr. Mitch Bryson

If you make use of this repo, please cite:

Windrim and Bryson. [Detection, Segmentation, and Model Fitting of Individual Tree Stems from Airborne Laser Scanning of Forests Using Deep Learning](https://www.mdpi.com/2072-4292/12/9/1469). Remote Sensing 12.9 (2020).

**Readme contents**:

- [Forest 3D App: inventory from pointcloud data.](#forest-3d-app-inventory-from-pointcloud-data)
  - [Setup/Installation](#setupinstallation)
    - [Setting Up Forest 3D App](#setting-up-forest-3d-app)
    - [Docker Installation](#docker-installation)
    - [Installing Darknet (Optional)](#installing-darknet-optional)
  - [Quickstart](#quickstart)
  - [How to use the Forest 3D libraries](#how-to-use-the-forest-3d-libraries)
    - [Reading data](#reading-data)
    - [Ground Characterisation and Removal](#ground-characterisation-and-removal)
    - [Tree Detection and Delineation](#tree-detection-and-delineation)
    - [Inventory](#inventory)
    - [Stem Segmentation](#stem-segmentation)
  - [Pre-trained models](#pre-trained-models)
    - [Tree detection models](#tree-detection-models)
    - [Stem segmentation models](#stem-segmentation-models)
  - [How to access pre-trained models](#how-to-access-pre-trained-models)
  - [How to train your own detection model](#how-to-train-your-own-detection-model)
    - [Generating training rasters](#generating-training-rasters)
    - [Labelling rasters with bounding boxes](#labelling-rasters-with-bounding-boxes)
    - [Training a yolov3 model with Darknet](#training-a-yolov3-model-with-darknet)
  - [How to train your own segmentation model](#how-to-train-your-own-segmentation-model)
    - [Labelling the training data](#labelling-the-training-data)
    - [Training a segmentation model](#training-a-segmentation-model)
  - [Detection Debugging Tips](#detection-debugging-tips)
  - [Inventory Results for Several Datasets](#inventory-results-for-several-datasets)
  - [Functions for high-resolution (TLS/Hovermap) tree detection](#functions-for-high-resolution-tlshovermap-tree-detection)
  - [Related publications](#related-publications)

![Alt text](media/drawing6.png?raw=true "Tree delineation using Forest 3D is designed to be flexible to the type of pointcloud and scale from plots to large compartments.")

## Setup/Installation

### Setting Up Forest 3D App

If you are using Ubuntu or Mac OS, clone the repository:

```
git clone https://github.com/lloydwindrim/forest_3d_app.git
```

Recommendation: setup and run within a virtual environment to isolate the python packages from those on your system. To create a blank python3 virtual envrironment:

```
cd forest_3d_app
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

### Docker Installation

you can also run the code in a docker container. If you have docker installed, you can build the docker image with:

```
docker build -t forest3d .
```

Then run the container with:

```
docker run -it -v <local-code-directory>:./3d_forest forest3d bash
```

If you are using Windows, download the repository from Github, and then install the python packages: numpy, scipy, scikit-learn, scikit-image, matplotlib, laspy, plyfile, opencv-python. Check requirements.txt for the correct versions of each.

**Stem Segmentation**

If you want to use the stem segmentation functionality, run the following command instead:

```
pip install -r requirements_seg.txt
```

This installs the packages needed for Forest3D with the addional packages for tensorflow and its dependecies that are needed for stem segmentation. By default, tensorflow 2.X is installed, but you can also use the older tensorflow 1.X if you want - both will work. The code is designed to check which version you have and import the legacy libraries if you are using version 1.X.

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

## Quickstart

Set your python interpreter to the virtual environement just created (if using the console just activate the virtual environment as above).

Create the following folders if you don't already have them (recommended in the forest_3d_ap root directory):

- **outputs**: this will store anything the script generates (e.g. ground mesh, delineated pointlcoud, inventory csv, etc.)
- **models/detection**: this holds the detection models. The subfolder /detection is recommended in preparation for future /stem_segmentation models.

Note: you don't need to use these exact names. Make sure models/detection has a model in it.

Next, navigate to the example detection scripts. From the root:

```
cd src/example_detection_scripts
```

Choose a script you want to run (based on the dataset). Edit three paths in the script:

- path to the pointcloud (e.g. a las file)
- path to the outputs folder (which you created before)
- path to a detection model (e.g. 'path/to/models/stem_segmentation/tumut1')

Then you can run the script, for example:

```
python tumut_detect1.py
```

**Adding in segmentation**

If you want to run a full pipeline which does detection and segmentation, then navigate to the example full pipeline scripts. From the root:

```
cd src/example_full_pipeline_scripts
```

modify the same three paths as in the detection case, with the addition of:

- path to the segmentation model (e.g. 'path/to/models/detection/model1')

Then run the script:

```
python tumut_pipeline.py
```

## How to use the Forest 3D libraries

There are several Forest 3D libraries you can use by importing them into a python environment. Make sure your python interpretter is set to be the one within the virtual environment. Then in a python script in your own project directory, you can import the libraries by first adding the path of the Forest 3D app source code:

```
import sys
sys.path.insert(0, 'path/to/forest_3d_app/src')

from forest3D import lidar_IO,ground_removal,treeDetector,detection_tools,processLidar,inventory_tools,stemSegmenter
```

There are libraries for:

- reading (and writing) pointcloud data into (and out of) python
- removing, exporting and importing the ground
- rasterising pointclouds
- detecting and delineating trees
- segmenting stem and foliage points
- useful classes for representing pointclouds
- computing inventory

There are also a range of useful tools and utilities for doing a variety of other things.

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

- the intensity is read in as raw values. You will see in some of the scripts that we scale the intensity channel/column to be between 0 and 1. This is done by dividing all values in that column by the upper bound for intensity (50,000)

```
xyz_data[:, 3] /= 50000.0
```

- for the colour channels, by setting convert_colours=True in lidar_IO.readFromLas() the values are read in between 0 and 255. We then scale colour channels in our array to be between 0 and 1:

```
xyz_data[:, 3:] /= 255.0
```

### Ground Characterisation and Removal

![Alt text](media/gr.png?raw=true "Characterisation of the ground with a mesh surface and removal of points within a threshold above the ground.")

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

- If a path string is given to _proc_path_, it will output the ground as a ply mesh to that path. If no path is given, nothing will be output. It is useful to establish an output directory where you will store all of the outputs from running Forest 3D functions (such as the ground mesh).
- _thresh_ is a single float which gives the distance above the ground mesh under which points are removed. This can be set differently for different tasks, but its primary purpose is to remove the ground and ground vegetation to make other tasks easier. For example, if you are detecting mature pine trees, you should set this to at least 2 metres (note the units are the same as the pointcloud array units) so that ground vegetation is removed and not falsely detected as a tree. If you want to measure dbh, you will want a seperate ground-removed-pointcloud with a threshold of 1 metre. If you are detecting seedlines you could set this to 0.5 metres.
- _offset_ is just a list of 3 values (x, y and z) to offset the pointcloud with (default [0,0,0]). This should only be used if you notice some array calculations on very large numbers are becoming unstable. Note, if you use a non-zero offset, the returned array will have the offset values. When saving outputs later, you will have the chance to remove the offset so the saved data has the original coordinate frame.

If you have intensity or colour for each pulse return information, you call the function with the additional _returns_ parameter:

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

- yolov3.cfg - the configuration file for the yolov3 detection network. This file describes the neural network architecture.
- yolov3.weights - the weights that were learnt when the yolov3 network was trained.
- raster_config.json - the raster configuration. The pointcloud must be rasterised in the same way as the rasters that were used to train the yolov3 detector. This config file rasterisation parameters so the process can be replicated.

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

- detector_addr - a string with the path to your detector model folder (the same one which has your raster_config.json).
- xyz_data_gr - your x,y,z pointcloud array (i.e. the one you removed the ground points from). Should be Nx3.
- ground_points - array output by ground_removal.load_ground_surface(). Only necessary if using the 'canopy_height' raster layer
- widowSize - size (metres) of window that slides through pointcloud and extracts out a crop to be rasterised.
- stepSize - the stride of the sliding window (metres). Should be less than the window size.

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
labels = detection_tools.label_pcd_from_bbox(xyz_data, boxes, yxyx=True)
```

Some other arguments of RasterDetector.sliding_window() that can be tuned are:

- classID - integer that specifies the class of interest. Default is 0. Check obj.data to see the class ID's of different classes
- confidence_thresh - the threshold on bounding box confidence. Default 0.5. Setting it closer to 0 finds more bounding boxes (possibly of lower quality), setting closer to 1 finds fewer.
- overlap_thresh - a threshold for removing overlapping boxes. This is an important parameter to tune, based on the distance between trees. Default 5 for mature Tumut pines. Can set to smaller for other datasets (e.g. for seedlings use overlap_thresh=1, Rotorua hovermap use overlap_thresh=2)

You can output the delineated pointcloud as a ply file for visualisation using:

```
lidar_IO.writePly_labelled(os.path.join(output_dir,'detection.ply'),xyz_data_gr,labels,offset=[0,0,0,0])
```

xyz_data_gr should have the same number of rows as labels. output_dir is the path to your output directory. Offset is optional here (default [0,0,0,0]). If you used an offset before to avoid unstable calculations, then put the offset here to revert the pointcloud to the correct coordinate space.

Alternatively, you can output the pointcloud as a las file, where the tree ID is stored in the point source ID las field:

```
lidar_IO.writeLAS(os.path.join(output_dir,'detection.las'),xyz_data_gr,labels,offset=[0,0,0])
```

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

### Stem Segmentation

![Alt text](media/segmentation.png?raw=true "Stem point segmentation.")

If you included the stem segmentation functionality in your setup of Forest3D then you can use the library for segmenting points for an individual tree in stem and foliage points.
The process this library uses can be broken down into a few key steps: pointclouds are voxelised, passed through a trained 3D neural network that segments the voxels into stem and foliage classes, then the labelled voxels are converted to low resolution pointclouds, and finally a KD-tree is used to upsample the labelled pointcloud to match its original resolution.

The library can be imported using:

```
from forest3D import stemSegmenter
```

The stemSegmenter library is similar to the treeDetector library in that it is class-based. The main class is called VoxelStemSeg(), constructed with a config json file from a stem segmentation model as follows:

```
segmenter_addr = ''path/to/segmentation/folder''
with open(os.path.join(segmenter_addr, 'net_config.json')) as json_file:
    config_dict = json.load(json_file)
stem_model = stemSegmenter.VoxelStemSeg(**config_dict['model_dict'])
```

The stemSegmenter.VoxelStemSeg object you create (here called stem_model) requires that the pointcloud passed into it be delineated, in the form of a list of pointcloud arrays, where each pointcloud array in the list corresponds to the points of a single tree. You might already have the files of delineated trees, which you could read in using the lidar_IO library. Or you might have delineated trees using the treeDetector library earlier in your code. You could use the **labels** array output by the sliding_window() method of your treeDetector.RasterDetector object to produce the list of pointclouds:

```
xyzr_list = []
for i in list(np.unique(labels)):
    if i > 0:
        xyzr_list.append(np.hstack((xyz_data_gr[labels==i,:],intensity_gr[labels==i])))
```

Remember to skip points labelled with 0 as this is the background class. Once you have the list of tree arrays, call the predict() method of your VoxelStemSeg object:

```
seg_list = stem_model.predict(xyzr_list,segmenter_addr)
```

with arguments:

- xyzr_list - list of Nx3 or Nx4 arrays depending on whether your segmentation model uses return intensity information or just x,y,z coordinates. Check this in the net_config.json file of the segmentation model folder by looking at the boolean of the keyword "isReturns". True means it uses return intensity so requires arrays to be Nx4 and False means no return intensity is used so arrays can be either Nx3 or Nx4.
- segmenter_addr - a string with the path to your segmentation model folder (the same one which has your net_config.json).
- batchsize - integer, the number of trees processed in parallel (default=5).
- returnSplit - boolean, if True the pointcloud and labels are seperated (returns two instances instead of one). Default False.
- dist_threshold - float (default 0.5). During the KD-tree process, if the nearest low res label to a high res point is beyond dist_threshold (metres) then the high res point is removed. This results in the labelled pointcloud having slightly fewer points. If you want to preserve all points, set dist_threshold=float("inf"). Do this if you want the intensity array to align with the labels array.

The output of the predict function is a list of Mx4 arrays (if returnSplit=False), where each array corresponds to a separate tree. The first three columns are x,y and z coordinates. The fourth column is the label (1-foliage, 2-stem).
Note: if dist_threshold is finite then the number of points M is different to N for correponding trees because some noisy points are removed during the segmentation process. Therefore you should use the Mx4 array as your new x,y,z coordinates (so the points align with the labels). If dist_method is infinite then M==N.

If you have a GPU with a reasonable amount of VRAM than you can use the batchsize parameter to increase the number of trees processed in parallel to speed up processing time. Likewise, if the default is too big for your computer's memory, use a smaller batchsize. With 11Gb of VRAM I could use a batchsize of 20:

```
seg_list = stem_model.predict(xyzr_list,segmenter_addr,batchsize=20)
```

You can output the labelled pointclouds as asc files and view them in cloudcompare using:

```
for i in range(len(seg_list)):
    lidar_IO.writeXYZ_labelled(os.path.join(output_dir,'labelled_%i.asc'%(i)), seg_list[i][:,:3],labels=seg_list[i][:,3], delimiter=',')
```

Or alternatively you can output them as las files, where the stem label falls under the point source ID field:

```
for i in range(len(seg_list)):
    lidar_IO.writeLAS(os.path.join(output_dir,'labelled_%i.las'%(i)), seg_list[i][:,:3], labels=seg_list[i][:,3])
```

**Segmentation uncertainty**

If you want the segmentation labels as well as a measure of the uncertainty of those labels, you can use the predict_uncertianty() method instead:

```
seg_list = stem_model.predict_uncertainty(xyzr_list,segmenter_addr)
```

This uses the regression variant of the MC dropout technique to output a list of Mx5 arrays (if returnSplit=False), where column 5 is the label and column 4 is the certainty of that label. Note, by default this process takes 5 times as long (the n_passes argument dictates this).

See the script example_segmentation_scripts/tumut_uncertainty.py for an example of how the uncertainty can be used and output in several different ways.

## Pre-trained models

Several models have been trained on forest pointcloud data and are available for use. The table describes each model. The appropriate model for your dataset should be the model trained on data most similar to your own dataset. Make sure you have return intensity or colour information for each point if the model requires it.

Acknowledgements to Interpine, UTAS, Forect Corp and the other project partners responsible for collecting the data used to train the models.

### Tree detection models

| Model name |       Dataset used for training        |     Location |   Species/Age |                 Sensor |                                                                                                      Description |
| ---------- | :------------------------------------: | -----------: | ------------: | ---------------------: | ---------------------------------------------------------------------------------------------------------------: |
| tumut1     | V1_Scanner1_161011_220153_crop001.las  |    Tumut NSW |   Mature Pine | VUX-1 ALS (helicopter) |                                           High res pointcloud. Only trained on x,y,z data (no return intensity). |
| tumut2     | V1_Scanner1_161011_220153_crop001.las  |    Tumut NSW |   Mature Pine | VUX-1 ALS (helicopter) |                                                 High res pointcloud. Trained on x,y,z and return intensity data. |
| hvp        |         saltwater_31B_1_2.las          |          HQP |   Mature Pine |                ? ALS ? |                                                   Low res ALS. Only trained on x,y,z data (no return intensity). |
| utas       |  transect_large_vis_dense_down5cm.las  |     Tasmania | Pine Seedling |   Drone Photogrammetry | Dense photogrammetric pointcloud. Trained on x,y,z with 2 principle components of red,green,blue for each point. |
| hovermap   | Interpine_02_Output_laz1_2_down5cm.las | Rotorua, NZL |   Mature Pine |      Hovermap backpack |               High res and low res under-canopy data similar to TLS. Trained on x,y,z and return intensity data. |
| hovermap2  | Interpine_02_Output_laz1_2_down5cm.las | Rotorua, NZL |   Mature Pine |      Hovermap backpack |                           High res under-canopy data similar to TLS. Trained on x,y,z and return intensity data. |

If '\_down5cm' is appended to the filename as in xxx_down5cm.las, it means the original dataset xxx.las was downsampled to 5cm resolution.

### Stem segmentation models

| Model name |       Dataset used for training       |  Location | Species/Age |                 Sensor |                                                            Description |
| ---------- | :-----------------------------------: | --------: | ----------: | ---------------------: | ---------------------------------------------------------------------: |
| tumut1     | V1_Scanner1_161011_220153_crop001.las | Tumut NSW | Mature Pine | VUX-1 ALS (helicopter) |       High res pointcloud. Trained on x,y,z and return intensity data. |
| tumut2     | V1_Scanner1_161011_220153_crop001.las | Tumut NSW | Mature Pine | VUX-1 ALS (helicopter) | High res pointcloud. Only trained on x,y,z data (no return intensity). |

For each segmentation model, points predicted to be foliage are labelled with 1, and points predicted to be stem are labelled with 2. Labels are floats.

## How to access pre-trained models

Pre-trained models must be downloaded separately from the main Forest3DApp repository: you can download the model through this link:

https://drive.google.com/file/d/1E4gHTo2Zdae51JDXHXtwgjGI2wtn1yN-/view?usp=sharing

Once downloaded, you will need to uncompress the zip file and place the "models" directory inside the "Forest3DApp" directory on your local copy.

## How to train your own detection model

Training your own detection model comprising three main steps: generating raster images, labelling raster images with bounding boxes and training a model using darknet.

To do this, you will need some have some raster data with labels and also have installed Darkent (see [Installing Darknet](#installing-darknet) above).

### Generating training rasters

The python scripts in **/gen_raster_scripts** can be used to generate raster jpg's from pointcloud data for training detection models. There are a few different ways to run one of these scripts depending on the kind of data you have:

- you have several pointcloud crops (not limited to circles) you wish to convert to rasters. You might have cropped these yourself from a larger pointcloud. For example gen_training_rasters_tumut1.py converts pointcloud crops saved as .asc into rasters.
- you have a single large pointcloud and have a csv file with the locations of circle centers and radii. For example see gen_training_rasters_hovermap.py, where training_locations.csv contains the places to crop the pointcloud. This is the simplest way to generate rasters.
- you have a single large pointcloud and have several csv's, one for each circle, that contain the cropping circle center and radius information. This is useful if you have field measurements for a number of plots. Each plot will have its own csv, where the first row specifies the location and size of the plot. See gen_training_rasters_hvp.py for an example, where each plot has its own training_stems_xxx.csv.

For the latter two methods, the detection_tools.circle_crop() function is useful:

```
xyz_crop_gr = detection_tools.circle_crop(xyz_data_gr,x_centre,y_centre,radius)
```

Note: the raster generation process includes ground removal. A similar threshold for removing points above the ground should be used in the detection script.

The key output here is raster jpg's. As can be seen in the scripts in gen_raster-scripts/ there are several ways to do this. Whichever method you use, there are just a few important things which must always be done:

- output a raster config file. This is so that the rasterisation process you use to generate the training rasters can be replicated during detection. Put the raster config file in the relevant detection model folder.
- save, output or record any pre-processing steps you do. For example, if you scale the intensity or colour between 0 and 1, you must also do this in the detection script. Another example, in gen_training_rasters_utas.py the principle compoenents of the colour channels are used to generate the training rasters. The function for transforming colour to principle components is saved as a pickle so that it can be used in the detect script (as opposed to calculating the principle components during detection using a slightly different function based on different data).
- you must make sure each pointcloud crop will fit within the raster (see discussion on window size, grid size and resolution in [Tree Detection and Delineation](#tree-detection-and-delineation)). Calculate the maximimum size your pointcloud can be based on the grid size and resolution, and then give yourself an additional buffer as sometimes the pointcloud doesnt sit in the exact centre of the raster.

**Raster Configuration**

The same treeDetector.RasterDetector object used for detection is also used in the raster generation process:

```
config_dict = {'raster_layers': ['vertical_density', 'canopy_height'], 'support_window': [11, 7, 1],
               'normalisation': 'rescale+histeq','doHisteq':[False,True],'res': 0.1, 'gridSize': [600, 600, 1000]}
rasterMaker = treeDetector.RasterDetector(**config_dict)
```

The config dictionary contains the parameters of the raster. This config is saved as raster.config and used in the detection script to replicate the rasterisation process.

Once the object has been created (e.g. rasterMaker in the above example) it can be used to rasterise a pointcloud crop:

```
raster = rasterMaker.rasterise(xyz_crop_gr,ground_pts=ground_pts)
```

You can then save each raster as a jpg with:

```
plt.imsave(os.path.join(output_dir,'raster_filename.jpg'), raster)
```

Finally, make sure to save the raster config file:

```
with open(os.path.join(output_dir,'raster_config.json'), 'w') as outfile:
    json.dump(config_dict, outfile)
```

The treeDetector.RasterDetector object created has a specific way of rasterising pointcloud data to a three channel colour image, which is given by its construction parameters (populated by the config_dict):

- raster_layers - a list of between one and three strings, where each string is a per channel rasterisation approach E.g. ['vertical_density', 'canopy_height', 'mean_colour1']. It is sufficient to only put one name instead of three if the 'cmap_jet' normalisation approach is used as it applies to all three channels. If another normalisation approach is used, then you can put 1-3 methods. Non-specified raster channels (e.g. you specify one or two methods and dont use 'cmap' normalisation) will be information-less rasters of uniform values (i.e. 0.5). A description of each raster layer is given below.
- support_window - a list of ints which specify the size of neighbourhoods used to rasterise points (default 1 - no neighbourhood used). This is useful for sparse pointclouds such as low res ALS.
- normalisation - a single string specifying the normalisation approach. Options are 'rescale', 'rescale+histeq' and 'cmap_jet'. 'rescale' scales values between 0 and 1, whilst the 'histeq' addition equalises the histogram of values. 'cmap' scales a single raster to a three-channel image using the jet colourmap.
- doHisteq - list of booleans relating to normalisation. Only applicable if using 'rescale+histeq' normalisation. This allows you to specify which of the three layers should have histogram normalisation.
- res - single float specifying the resolution of each raster grid cell (metres/cell).
- gridSize - list of 3 ints specifying the number of cells in each axis of the grid (x,y,z). Even though a raster image layer is 2D, the raster uses a third dimension (i.e. z-axis) to compute the raster values (e.g. the density along z columns).

The raster layers available are:

- 'vertical_density': the density of points along the z-axis. Useful if stems are visible in the scan as they produce a distinct density pattern for identifying a tree.
- 'max_height': captures the maximum height of points at locations in the xy-plane. Useful if canopy is visible. Heights are not normalised by the ground.
- 'canopy_height': similar to 'max_height' but normalised by the ground. I.e ground normalised max height for locations in the xy-plane.
- 'mean_colour1', 'mean_colour2', 'mean_colour3': the average colour along the z-axis (i.e. a value for each xy-location). The colour is either the return intensity or other colour of the point (e.g. r, g, b, principle component, etc.). If each point has multiple colour channels, then the number prefixing 'mean_colour' specifies the channel used.
- 'max_colour1', 'max_colour2', 'max_colour3': similar the 'mean_colourX' but taking the max colour along the z-axis. This is similar to vertical occupancy, but values are not restricted to 1 or 0.

The res parameter has several implications. Often object detectors have a hard time detecting small objects in images. By lowering the res parameter (increasing the resolution) you can increase the size of the objects in the image, making them easier to detect. There is however a trade-off in speed to doing this. The smaller the resolution, the smaller the amount of coverage the sliding window has in the pointcloud. This makes detection slower. Lowering the res parameter also increases the sparsity of the point representation in the raster which could make it harder to detect objects. However, you can get around this by increasing the support window size.

The detection speed is also linked to the number of raster layers that have to be created. The more raster layers that have to be created, the slower the detection will be. The fastest detector would only have a single raster layer.

**Leveraging field measurements to help with the labelling**

If you have some field data, such as differential gps measurements of tree locations, then it is possible to make the bounding box labelling step easier for yourself by generating rasters with annotation markings on them. This removes much of the ambiguity from the labelling process as you just have to draw boxes around each marking. The bounding box label files (more on them in the next section) are only connected to the raster jpg's by filename (e.g. raster_0.jpg will be matched with raster_0.xml label file). This means you can generate rasters with annotations on them to help you create the label xml files, and then just re-generate the rasters without the annotations on them for training the detector (do not train on the annotated rasters). The rasters without the annotations will be linked to the label files as long as they have the same name.

An example script for generating rasters with annotations is **gen_training_rasters_hvp_anno.py**, which is similar to the raster generating script **gen_training_rasters_hvp.py**, but it generates rasters with dots on them indicating tree locations marked out in the field. The script assumes field measurements as a csv for each plot. Another script, **gen_training_rasters_utas_anno.py**, assumes all field measurements are in a single csv.

Use a function to convert field locations from global pointcloud coordinates to raster coordiantes. With the raster coordinates of object locations you can highlight a set of pixels in the raster around that location or otherwise.

```
raster,centre = rasterMaker.rasterise(xyz_crop_gr,ground_pts=ground_pts,returnCentre=True)
raster_coords = treeDetector.pcd2rasterCoords(gt_coords, config_dict['gridSize'], config_dict['res'],centre)

```

Here gt_coords is an array of x and y locations in the pointcloud that were marked out in the field. raster_coords are the row and column in the raster of each location.

![Alt text](media/drawing7.png?raw=true)

**Re-using labels with different raster configs**

For similar reasons as to why you can generate rasters with annotations to help with labelling, if you already have a set of raster jpg's and corresponding label xml files, but you want to try out training with a different raster config, it is possible to re-use the same label files. As long as the names match, and the new raster configuration does not change the position of any of the trees (e.g. do not change the grid size or resolution), then the xml files will still give the correct labels. This is because the xml files only describe the position and class of objects in rasters and record nothing about the raster itself. This gives you the flexibility to change the raster layers, support window and normalisation method without having to re-label anything!

### Labelling rasters with bounding boxes

Once you have some rasters, you can label them with bounding boxes. We suggest using a tool called [labelImg](https://github.com/tzutalin/labelImg), but you can use any tool you want as long as it can output the labels as XML files in PASCAL VOC format.

Once you have installed labelImg, point it to your directory of rasters and a directory where you wish to save you bounding box labels. Then start drawing boxes around ALL objects of interest in each raster. The detection script is designed for three classes:

- **tree**
- **shrub** - ground vegetation (there shouldn't be too much of this because we remove alot of it).
- **partial** - these are trees that get cut in half because they lie on the boundary of where the pointcloud is cropped. It is important to detect these as a seperate class to trees so that they are ignored and the full detetion (which we will get if our sliding window overlaps a bit) is used instead.

If you are using the detection script _as-is_, it is very important to use these class names when labelling (don't make a spelling mistakes). If you want to speed up your labelling, check out the hotkeys at [labelImg](https://github.com/tzutalin/labelImg). Also, you might find it handy to look at the data in 3D as well to get a more complete perspective of the raster (just open the pointcloud with your 3D viewer) .

Make sure the bounding box label files (the .xml files) have the same name as their corresponsing image (.jpg) file. This should happen by default if you use the labelImg tool.

### Training a yolov3 model with Darknet

Make sure you have Darknet installed. If you are training your first model, from forest_3d_app/src/darknet_scripts copy setup.sh and the custom_setup/ folder to your Darknet root. Then run:

```
./setup.sh
```

This copies two python scripts and four config files to where they need to be in the Darknet root. It will also download some weights pre-trained on Imagenet to initialise the convolutional layers of yolov3. These are downloaded to a folder called 'models'. The config files specify the architecture and training configuration for training and prediction. Config files are available for both the yolov3 and yolov3-tiny architectures. The python scripts are for converting the bounding box xml labels from PASCAL VOC format to YOLO format, and splitting data into test and training data.

Open up the shell script auto_setup_yolo.sh (in forest_3d_app/src/darknet_scripts). This script does all of the heavy lifting for training your detection model. Open it with a text editor and edit the following paths in the header:

- darknet_location: the address of the darknet directory root
- img_location: the address of the folder containing your raster jpg's
- anno_location: the address of the folder containing your bounding box label xml files

Then run the shell script in the console:

```
./auto_setup_yolo.sh
```

This will create a folder called tmp_yolo which has everything needed for training. The console should also start displaying training information such as losses and the epoch number (corresponding to batch number in the config file, where max_batches is the total number of epochs that will run). This will stop when it is finished training. Depending on your GPU, this might take 12 hours to several days. It took about 24 hours to run 200,000 batches (config default) on my Nvidia 1080ti GPU.

When it is finished, you should have a folder called tmp_yolo/deploy. This folder has three out of the four files you will need to do detection (the other being raster.config): yolov3.cfg, yolov3.weights and obj.names. Move these files to a folder a new detection model folder in your project directory. The tmp_yolo/backup folder also has some .weights model files saved at earlier epochs. You could experiment with these models as well.

To train future models, you don't need to run setup.sh again. Just modify the auto_setup_yolo.sh script and/or modify the custom config files in the cfg folder of the darknet directory. Make sure to rename your tmp_yolo folder created during the last training session of you want to keep it as it will be overwritten otherwise.

**Using yolov3 instead of yolov3-tiny**

By default the auto_setup_yolo.sh is actually set to use a variant of yolov3 called yolov3-tiny. It is a slightly smaller architecture that runs faster with a small performance trade-off. You can quite easily switch to the normal yolov3 architecture - you just have to modify the config files that are pulled across by the auto_setup_yolo.sh script:

```
cp cfg/yolov3-custom.cfg tmp_yolo/cfg/yolov3-tiny-custom.cfg
cp cfg/yolov3-custom-test.cfg tmp_yolo/cfg/yolov3-tiny-custom-test.cfg
```

They will still have 'tiny' in their name, but the contents will now reflect the non-tiny yolov3 architecture.

**Modifying the training configuration**

If you want to modify the training configuration in some way, than you need to do so using the .cfg files in the cfg folder of the darknet directory or in the auto_setup_yolo.sh script. The file yolov3-tiny-custom.cfg is used for training and yolov3-tiny-custom-test.cfg is used for prediction (i.e. detection).
To change the number of epochs, batchsize or subdivisions, do so in the training cfg file (e.g. yolov3-tiny-custom.cfg ).

An epoch is considered one training cycle through the entire training dataset. Therefore your batchsize and dataset size will dictate how many batches (or iterations) it takes to complete one epoch, and the number of epochs is a function of the number of batches (i.e iterations) and the batchsize.

- number of batches or iterations: this is set with max*batches. Note, whatever you put as max_batches, you should put steps as 0.8 * max*batches and 0.9 * max_batches.
- batchsize: set with the batch parameter. Default is 16. Make sure you look at the uncommented line in the training config (not the commented out line).
- subdivisions: this dictates how a batch is further split to train on (e.g. default subdivision of 2 with a batch size of 16 means 8 samples loaded into memory for backpropagation at a time). This parameter is important for memory limitations. If youre GPU doesn't have the memory to handle your batchsize, then you should increase this parameter to reduce the memory load.

I used max_batches=200,00 with typically small training sets of around 20 samples.

To change the training/validation split, do so in auto_setup_yolo.sh in the line that says:

```
python scripts/split_train_test.py -d "$img_location" -y "$darknet_location" -s "10"
```

(default is 10 meaning 90/10% training/validation split). The rasters used for training and validation are chosen randomly but get specified in tmp_yolo/train.txt and tmp_yolo/val.txt which get generated once you run auto_setup_yolo.sh.

**Training to detect different classes**

The scripts are setup to train a detector on 3 classes: "tree","shrub" and "partial". If you want to train a detector on different classes to these, there are a few things you must modify:

In auto_setup_yolo.sh:

- change num_classes=3 in the header
- modify the line to include your class names:

```
python scripts/voc_label_custom.py -d "$anno_location" -c "tree,shrub,partial"
```

The order of classes here establishes the class indexes. This determines the order they appear in they obj.names generated file.

Inside both config files (training and prediction) make the following changes:

- classes=3
- in the [convolutional] block before each [yolo] block, change filters=(num_classes + 5) \* 3 (note: evaluate the expression). E.g. for 3 classes this is filters=24

In yolov3-tiny there are 2 locations to make each of these changes in each config file. In yolov3, there are 3 locations for each change. These changes modify the architecture for the different number of classes.

## How to train your own segmentation model

The two main steps to training a segmentation model are to curate a labelled training dataset and then run a script which trains a model.

### Labelling the training data

A training sample for a segmentation model is a single tree with the ground removed and stem, foliage and clutter components labelled. All of this can be done using the segmentation tool of [CloudCompare](https://www.danielgm.net/cc/):

- crop out a single tree
- crop the ground out
- crop the tree into numerour segments, where each segment is a single class. E.g you might have several segments for folaige and several segments for stem
- for each class, save all segments as asci files with the class name in the filename (i.e. foliage, hgStem, lgStem, clutter) into a folder for each tree. hgStem stands for high-grade stem and refers to the lower part of the stem which does not have any branches growing out of it. lgStem stands for low grade stem and is the upper portion of the stem.
- run the python script **collate_cc_segments.py** for several tree folders. This will create a single asci file with the x, y, z, returns and labels columns, which can be used to train a stem segmentation model.

The code is currently configured for the labels:
0-foliage, 1-lower stem (hgStem), 2-upper stem (lgStem), 3-clutter. The lower stem and upper stem are collapsed into one class, so you do not need to label these separately (just label a single stem class as either 1 or 2). Likewise, the clutter class gets collapsed with the foliage class. So you do not need to label it as a seperate class, just label clutter as foliage (0).

You also do not have to include the return intensities in this process. But if the information is available, it is helpful to include incase you change your mind later (they can still be registered with the labelled data but it is a more complicated process).

### Training a segmentation model

The python scripts in /train_segmenter_scripts.py are example scripts for training stem segmentation models on the Tumut dataset. **train_tumut1_voxnet.py** uses the return intensity information, whereas **train_tumut2_voxnet.py** does not.

The scripts read in files for individual labelled trees into python and build a list of arrays, where each tree is has own array. Then the indexes used for training and validation are established based on a specified number of validation samples (e.g. 10).

A dictionary with the model and training configuration parameters is created. This is actually setup as a dictionary of three sub-dictionaries: model_dict, train_op_dict and train_model_dict. model_dict has to do with the architecture and is therefore important for doing inference later on (to make sure the inference network replicates the training network). train_op_dict and train_model_dict are both concerned with training the model and therefore are not needed later on for inference. train_op_dict is used to create the training nodes in the tensorflow graph (e.g for the optimiser). train_model_dict contains paramters like the number of epochs and path of where the model should be saved.

model_dict:

- input_dims - list of 3 ints, the size of the voxel grid for each tree
- res - single float or list of 3 floats, the size of the voxels (in the same units as the pointcloud data). These apply to the x,y,z axes. If a single float is given, the same value is applied to all three axes.
- isReturns - boolean for whether or not to use return intensity information
- nClasses - for the model, this should be the number of classes of interest plus 1 for background. E.g. 3 for stem, foliage and background. If you use a number other than 3, changes will have to be made in processLidar.processPC to modify the occupancy grid that gets created.

train_op_dict:

- opt_method - string, the optimiser used. Options are 'Adam' (default), 'SGD' and 'RMS'.
- lr - float, the learning rate (default 0.0001)
- decay_rate - float, fraction by which lr decays. Default None.
- decay_steps - int, epoch cycle for decaying lr at decay_rate.
- piecewise_bounds - list of ints, specific epoch numbers to change lr to be piecewise_values.
- piecewise_values - list of floats, new lr values to happen at piecewise_bounds epochs

train_model_dict;

- nEpochs - int, number of cycles through the training dataset
- train_bs - int, batchsize for training
- val_bs - int, batchsize for validation
- augment - boolean, whether to use data augmentation (random rotations or trees about z-axis) default False
- numAugs - int, number of augmentations to do for each sample (default 3). Only applicable if augment=True
- keep_prob - float, rate at which neurons are kept active during dropout (training only). Default 1.0, which keeps all neurons.
- save_epochs - list of ints, which epochs to save the model at
- save_addr - string, path to folder for saving the model files
- save_figure - int, epoch update rate of figure which plots the losses (default 1)

These three sub-dictionaries are saved within a meta config dictionary and output as a net_config.json file within the segmentation model folder. This net_config.json file is necessary for using the model in the future for inference. It is also a handy record of the parameters used to train a particular model.

Once the config parameters have been specified, training iterator objects are created using the processLidar.iterator_binaryVoxels_pointlabels class. Iterators hold pointcloud data and have functions for calling the next batch of data to be trained on or used for validation. Separate iterator objects are created for training and validation. The construction arguments are the pointcloud, return intensities and labels as seperate lists of arrays, as well as the voxel dimensions, number of classes and batchsize specified in the config. Only pass in the return intensities if you want your model to use them (i.e. isReturns=True):

```
train_iter = processLidar.iterator_binaryVoxels_pointlabels([xyz_list[i] for i in train_idx],[label_list[i] for i in train_idx],
    returns=[intensity_list[i] for i in train_idx],res=model_dict['res'], gridSize=model_dict['input_dims'],numClasses=model_dict['nClasses']-1, batchsize=train_model_dict['train_bs'])
```

An object of the class stemSegmenter.VoxelStemSeg (the same used for prediction) is constructed with the model config parameters:

```
stem_model = stemSegmenter.VoxelStemSeg(**model_dict)
```

This builds the neural network as a tensorflow graph. Once constructed, you can setup the graph nodes needed for training by calling the train_op() method:

```
stem_model.train_op(**train_op_dict)
```

Finally, train the model using the train_model() method, passing in the iterator objects as arguments:

```
stem_model.train_model(train_iter, val_iter, **train_model_dict)
```

When training, you will see the progress printed in the console, and also a png figure which updates at least every epoch inside the folder you chose to save the model in. This figure shows the progression of the training and validation losses. Once the number of epochs reaches those in your save_epochs list (or the total number of training epochs), then a model will be saved as a seperate folder within your output folder.

When training is finished, put a copy of the net_config.json file inside that model folder, move the folder to a convenient location and then pass the path to that folder as the segmenter model address when you do prediction.

**Tips**

It is highly recommend you use a GPU for training. Because the data is composed of a batch of 3D voxels, the process is slow if you train on a CPU.

The models themselves actually have a small memory footprint because they are fully-convolutional as opposed to having dense layers. However, during training alot of memory is required for backpropagation. You will have to tune the batchsize and number of augmentations to make sure you have enough memory for training - remember the number of samples scales linearly with the number of augmentations+1. So if your batchsize is 5 and your number of augmentations is 3, then you are actually training on 5\*(3+1)=20 samples each batch.

## Detection Debugging Tips

Trying out your detector on a full pointcloud can take a couple of minutes, which is annoying if you have to tune parameters or even check if it works. Break up the process with some intermediate testing steps.

First you check for errors in the detector. This test isolates the detector from pointcloud processing errors. Use darknet to try out your detector on a raster image (the kind of one made by a gen_training_rasters python script) using the bash command:

```
./darknet detector test tmp_yolo/deploy/obj.data tmp_yolo/deploy/yolov3.cfg tmp_yolo/deploy/yolov3.weights raster_0.jpg
```

Check to see if the correct classes are being picked up, the bounding boxes are fitting the objects well, and there aren't too many false positive detections (commission errors) or false negative detections (omission errors). You might not have trained your detector for long enough, or maybe there was an error in the training process. This is the easiest point at which to catch most problems with the actual detector.

Once you know the detector works, the next test looks for errors in the pointcloud processing. Crop out a small section of the pointcloud using a tool like [CloudCompare](https://www.danielgm.net/cc/) and run that las file through your code. Look at the delineated pointcloud and see if it matches what you saw with the raster image test. If not, you might need to look at parameters like overlap_thresh in the sliding_window() method. There could also be a problem with your python script (e.g. data is being read in wrong, not rasterising properly). If that is the case you need to step through your code and check the outputs at each stage to make sure they are what you expect.

If you are getting lots of false positive detections from ground vegetation, experiment with a higher ground point removal threshold.

## Inventory Results for Several Datasets

The table below contains plot-level inventory outputs for several datasets, as computed by the Forest3D tools and models from this project. The name of the script in /example_detection_scripts is given so you can replicate the results.

| Dataset                          |                Filename                | Model used |             Script | No. trees detected | Typical height range (m) | Mean height (m) | Std height (m) |
| -------------------------------- | :------------------------------------: | ---------: | -----------------: | -----------------: | -----------------------: | --------------: | -------------: |
| Tumut (high res ALS)             | V1_Scanner1_161011_220153_crop001.las  |     tumut1 |   tumut_detect1.py |                523 |                    22-49 |           39.68 |           6.31 |
| Tumut FCNSW (high res ALS)       |    FCNSW_FertTrialAOI_VUX1_conv.las    |     tumut1 |   tumut_detect1.py |               1626 |                    19-33 |           25.99 |           2.55 |
| HVP (low res ALS)                |         saltwater_31B_1_2.las          |        hvp |      hvp_detect.py |               8693 |                    15-31 |           23.95 |           3.50 |
| UTAS seedlings                   |  transect_large_vis_dense_down5cm.las  |       utas |     utas_detect.py |               3618 |                0.39-3.69 |            2.18 |           1.47 |
| Rotorua Hovermap                 | Interpine_02_Output_laz1_2_down5cm.las |   hovermap | hovermap_detect.py |                811 |                        - |               - |              - |
| Rotorua Hovermap (high res crop) | Interpine_02_Output_laz1_2_down5cm.las |   hovermap | hovermap_detect.py |                 45 |                    32-42 |           38.93 |           2.06 |

Tree-level inventory outputs can be found in the /media folder as \*.csv files. These can be opened with excel, or visualised with the pointcloud using [CloudCompare](https://www.danielgm.net/cc/).

![Alt text](media/drawing5.png?raw=true "Height distribution of trees mapped spatially for several datasets.")

## Functions for high-resolution (TLS/Hovermap) tree detection

The Forest3DApp repository also contains a number of additional scripts and methods that can be used to process high-resolution pointcloud files (such as from TLS scanners or the Hovermap sensor) using a non-machine learning based pipeline (see further details in project final report). These scripts are located in the sub-folder "hovermap" in the "src" directory.

In addition to the required packages listed above, this script also makes use of the open-source package "Open3D" which you will need to install using pip. You can install this by typing "pip install open3d" in the command prompt.

To run this pipeline, open up the script "extract_stems_script.py" and edit (a) the path to the pointcloud file to process and (b) the path to the output directory to which processed data will be placed.

## Related publications

Some links to open-access publications that use these methodologies for processing 3D forestry data:

- Windrim and Bryson. [Detection, Segmentation, and Model Fitting of Individual Tree Stems from Airborne Laser Scanning of Forests Using Deep Learning](https://www.mdpi.com/2072-4292/12/9/1469). Remote Sensing 12.9 (2020).
- Windrim and Bryson. [Forest Tree Detection and Segmentation using High Resolution Airborne LiDAR](https://arxiv.org/abs/1810.12536). 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2019).

Links to other open-access forestry publications from our group:

- Windrim, Bryson, McLean, Randle, Stone. [Automated Mapping of Woody Debris over Harvested Forest Plantations Using UAVs, High-Resolution Imagery, and Machine Learning](https://www.mdpi.com/2072-4292/11/6/733). Remote Sensing 11.6 (2019).
- Windrim, Carnegie, Webster, Bryson. [Tree Detection and Health Monitoring in Multispectral Aerial Imagery and Photogrammetric Pointclouds Using Machine Learning](https://ieeexplore.ieee.org/abstract/document/9102401). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2020).
