'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy

    Useful functions relating to 2D and 3D object detection (helpful for images or pointclouds).

'''


import numpy as np
import pickle

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0]-windowSize[0]+1, stepSize): # want it to stop once a window touches the end
        for x in range(0, image.shape[1]-windowSize[1]+1, stepSize): # the plus 1 makes sure it does the last slide
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pad_img(image, stepSize, windowSize):
    row_excess = stepSize - ((image.shape[0]-windowSize[0])%stepSize)
    col_excess = stepSize - ((image.shape[1]-windowSize[1])%stepSize)
    padded_im = np.zeros((image.shape[0]+row_excess,image.shape[1]+col_excess,image.shape[2]),dtype=image.dtype)
    
    padded_im[row_excess/2:image.shape[0]+(row_excess/2),col_excess/2:image.shape[1]+(col_excess/2),:] = image
    return padded_im,row_excess,col_excess
    

def window2mosaic_coords(box, x,y, mosaic_shape, windowSize):
    return (np.array([((y+(box[:,0]*windowSize[0]))/mosaic_shape[0]),((x+(box[:,1]*windowSize[1]))/mosaic_shape[1]),((y+(box[:,2]*windowSize[0]))/mosaic_shape[0]),((x+(box[:,3]*windowSize[1]))/mosaic_shape[1])])).T
    
def ssd(a,b):
    return np.sqrt(np.sum((a-b)**2,axis=0))

def find_square_boxes(boxes,thresh):
    return np.absolute(1 - ((boxes[:,2] - boxes[:,0])/(boxes[:,3] - boxes[:,1])))<thresh
    

def find_unique_boxes(boxes,image_size,centres_dist):
    # for bounding boxes in percentages
    centres = (np.vstack([(boxes[:,2]+boxes[:,0])/2,(boxes[:,3]+boxes[:,1])/2]))
    centres = ( np.vstack([image_size[0]*np.ones((1,boxes.shape[0])),image_size[1]*np.ones((1,boxes.shape[0]))])) * centres
    most_square_idx = []
    for ii in range(centres.shape[1]):
        temp = ssd( (np.tile(centres[:,ii],(centres.shape[1],1))).T , centres )
        iii = [j for j, x in enumerate(temp<=centres_dist) if x] #35,150
        if len(iii)>1:
            iii = iii[np.argmin(np.absolute(1 - ((boxes[iii,2] - boxes[iii,0])/(boxes[iii,3] - boxes[iii,1]))))]
        else:
            iii = ii
        most_square_idx.append(iii)

    return np.unique(np.array(most_square_idx))

def find_unique_boxes2(boxes,centres_dist,prioritise='area'):
    # for bounding boxes in normal coords
    centres = (np.vstack([(boxes[:,2]+boxes[:,0])/2,(boxes[:,3]+boxes[:,1])/2]))
    centres = ( np.vstack([np.ones((1,boxes.shape[0])),np.ones((1,boxes.shape[0]))])) * centres
    most_square_idx = []
    for ii in range(centres.shape[1]):
        temp = ssd( (np.tile(centres[:,ii],(centres.shape[1],1))).T , centres )
        iii = [j for j, x in enumerate(temp<=centres_dist) if x] #35,150
        if len(iii)>1:
            if prioritise=='squareness':
                iii = iii[np.argmin(np.absolute(1 - ((boxes[iii,2] - boxes[iii,0])/(boxes[iii,3] - boxes[iii,1]))))]
            elif prioritise=='area':
                iii = iii[np.argmax((boxes[iii,2] - boxes[iii,0])*(boxes[iii,3] - boxes[iii,1]))]
        else:
            iii = ii
        most_square_idx.append(iii)

    return np.unique(np.array(most_square_idx))



def do_boxes_overlap(a,b):
    if (a[2]>=b[0] and b[2] >= a[0]) and (a[3]>=b[1] and b[3] >= a[1]):
        return True
    else:
        return False


def combine_two_boxes(a,b):
    return np.array([np.minimum(a[0],b[0]),np.minimum(a[1],b[1]),np.maximum(a[2],b[2]),np.maximum(a[3],b[3])])


def merge_boxes(boxes):
    unique_boxes = []
    for i in range(boxes.shape[0], 0, -1):
        for j in range(i - 1):
            if do_boxes_overlap(boxes[i - 1, :], boxes[j, :]):
                boxes[j, :] = combine_two_boxes(boxes[i - 1, :], boxes[j, :])
                break
            if j == (i - 2):
                unique_boxes.append(boxes[i - 1, :])
    return np.array(unique_boxes)

def extract3d_points(pcd,coords):
    # returns a pcd within the given bounds in the xy plane
    # coords [x_min, x_max, y_min, y_max]
    idx = (pcd[:, 0] > coords[0]) & (pcd[:, 0] < coords[1]) & (pcd[:, 1] > coords[2]) & (pcd[:, 1] < coords[3])
    if np.sum(idx) == 0:
        pcd_extract = None
    else:
        pcd_extract = pcd[idx, :]
    return pcd_extract

def extract3d_pointIdx(pcd,coords):
    # returns a pcd within the given bounds in the xy plane
    # coords [x_min, x_max, y_min, y_max]
    idx = (pcd[:, 0] > coords[0]) & (pcd[:, 0] < coords[1]) & (pcd[:, 1] > coords[2]) & (pcd[:, 1] < coords[3])
    return idx

def sliding_window_3d(pcd, stepSize, windowSize):
    # slide a window across a pcd in the xy plane
    # stepSize and windowSize is in meters [x y]
    for x in range(int(np.min(pcd[:, 0])), int(np.max(pcd[:, 0])),stepSize):
        for y in range(int(np.min(pcd[:, 1])), int(np.max(pcd[:, 1])),stepSize):
            # yield the current window
            yield (x, y, extract3d_points( pcd, [x,x+windowSize[0],y,y+windowSize[1]] ) )

def boundingBox_to_3dcoords(boxes_, gridSize_, gridRes_, windowSize_, pcdCenter_):
    # boxes = [ymin xmin ymax xmax] N x 4 (for N bounding boxes)
    # grid size np.array(( x , y )) in pixels (or list)
    # res - pixel size in meters
    # window size np.array(( x , y )) in pixels (or list)
    # pcd center - x and y coords of center (in meters) of pcd that was used to make raster where bounding boxes were found
    # returns boxes = [ymin xmin ymax xmax] in global 3d coords in the xy plane

    N = np.shape(boxes_)[0] # number of bounding boxes
    # convert from percentage to pixel coords
    bb_coord = np.transpose(np.vstack((boxes_[:, 0] * gridSize_[1], boxes_[:, 1] * gridSize_[0], boxes_[:, 2] * gridSize_[1],boxes_[:, 3] * gridSize_[0])))
    # center in OG
    bb_coord -= (np.tile(gridSize_[::-1], (N, 2)) / 2)
    # convert to meters
    bb_coord *= gridRes_
    # offset to match raster center with pcd center
    bb_coord += np.tile(pcdCenter_, (N, 2))
    # re-order so it is [ymin xmin ymax xmax]
    bb_coord = bb_coord[:,[1,0,3,2]]

    return bb_coord

def label_pcd_from_bbox(pcd,boxes,classes=None, yxyx=False):
    '''

    :param pcd: np.array of size Nx3 (x,y,z)
    :param boxes: np.array of size Mx4. box order is [x_min, x_max, y_min, y_max]. If want to use [ymin xmin ymax xmax], set yxyx=True
    :param classes: optional. Pass in stem labels if you want them delineated as well.
    :param yxyx: (bool)
    :return:
    '''

    if yxyx is True:
        boxes = boxes[:, [1, 3, 0, 2]]

    labels = np.zeros((np.shape(pcd)[0]),dtype=int)
    if classes is not None:
        class_labels = np.zeros((np.shape(pcd)[0]),dtype=int)
    for i in range(np.shape(boxes)[0]):
        idx = extract3d_pointIdx(pcd,boxes[i,:])
        labels[idx] = i+1
        if classes is not None:
            class_labels[idx] = classes[i]
    # dont permute the zero class (anything not in a bbox)
    label_array = np.hstack(( np.array((0)),np.random.permutation(range(np.max(labels)+1)[1:]) ))
    labels = label_array[labels]

    if classes is not None:
        return labels,class_labels
    else:
        return labels

def save_boxes(addr,boxes):
    # make sure addr has '.pkl'
    with open(addr, 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([boxes], f)

def load_boxes(addr):
    # make sure addr has '.pkl'
    with open(addr) as f:  # Python 3: open(..., 'wb')
        boxes = pickle.load(f)[0]
    return boxes

def IoU(og1,og2):
    intersection = np.sum( np.bitwise_and(og1 == 1, og2 == 1) )
    union = np.sum( np.bitwise_or(og1 == 1, og2 == 1) )
    if union == 0:
        return 0
    else:
        return float(intersection)/float(union)

def segment_precision_recall(og1,og2):
    # og1 - predicted
    # og2 - gt
    # returns precision, recall
    intersection = np.sum( np.bitwise_and(og1 == 1, og2 == 1) )
    if intersection == 0:
        return 0,0
    else:
        return float(intersection)/float(np.sum(og1 == 1)) , float(intersection)/float(np.sum(og2 == 1))

def evaluation_metrics(pred_flag,gt_flag):
    # flag is 1s for correct and 0s for incorrect
    # e.g. correct might have IoU > 0.5
    tp = float(np.sum(pred_flag)) # note: if no overlapping detections, then number of ones in pred_flag and gt_flag should be the same
    fp = float(np.size(pred_flag) - tp)
    fn = float(np.size(gt_flag) - np.sum(gt_flag))  # equivalent to np.length(gt_flag) - tp

    prec = tp/(tp+fp)
    recall = tp/(tp+fn)
    if (prec+recall) == 0:
        f1 = 0
    else:
        f1 = (2*prec*recall)/(prec+recall)

    return tp,fp,fn,prec,recall,f1


def rank_closest_pixels(im,target):

    cols = np.arange(0, np.shape(im)[1], 1)
    rows = np.arange(0, np.shape(im)[0], 1)
    cols_x, rows_y = np.meshgrid(cols, rows)
    if len(np.shape(im))==2:
        pts = np.transpose( np.vstack(( np.reshape(rows_y, [-1]),np.reshape(cols_x, [-1]),np.reshape(im, [-1]) ))  )
    else:
        pts = np.zeros(((np.shape(im)[0]) * (np.shape(im)[1]), 2 + np.shape(im)[2]))
        pts[:,0] = np.reshape(rows_y, [-1])
        pts[:, 1] = np.reshape(cols_x, [-1])
        for k in range(np.shape(im)[2]):
            pts[:,2+k] = np.reshape(im[:,:,k], [-1])
        
    dist = np.sum( ( pts[:, 2:] - np.tile(target,(np.shape(pts)[0],1)) ) ** 2 , axis=1)
    sorted_dist = np.argsort(dist)
    res = (pts[sorted_dist, :2]).astype(int)
    return res


def circle_crop(pcd,x,y,radius_max, radius_min=0,return_indices=False):

    idx_max = (((pcd[:, 0] - x) ** 2) + ((pcd[:, 1] - y) ** 2)) < (radius_max ** 2)
    idx_min = (((pcd[:, 0] - x) ** 2) + ((pcd[:, 1] - y) ** 2)) > (radius_min ** 2)
    idx = idx_max & idx_min
    if return_indices is False:
        return pcd[idx,:]
    else:
        return idx



