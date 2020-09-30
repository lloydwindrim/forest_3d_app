'''
    Author: Dr. Lloyd Windrim
    Required packages: numpy,plyfile,laspy

    Useful functions for reading and writing pointcloud data into and out of python. Covers a range of formats (e.g. ply,
    las, asc, csv, bin, off). It also has some functions for converting from voxels to points (helpful for visualising
    data).

'''

import numpy as np
import csv
from plyfile import PlyData, PlyElement
import glob
from laspy.file import File as Lasfile
import laspy.header as lasHeader

# also works for .asc [delimiter=' ',x=0,y=1,z=2]
def XYZreadFromCSV(filename,delimiter=',',x=3,y=4,z=5,label=None,returns=None):
	data = []

	with open(filename, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
		for row in spamreader:
			data.append(row)

	data = np.array(data)
	if (label is None) & (returns is None):
		return data[:,[x,y,z]].astype(float)
	elif (label is not None) & (returns is None):
		return data[:, [x, y, z, label]].astype(float)
	elif (label is None) & (returns is not None):
		return data[:, [x, y, z, returns]].astype(float)
	else:
		return data[:, [x, y, z, returns, label]].astype(float)


# accepts xyz data as a 2D array (points x dims)
def writeXYZ(filename,xyz_data,delimiter=' '):
	i = 0
	with open(filename, 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=delimiter,
			quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(np.size(xyz_data,0)):
			spamwriter.writerow(xyz_data[i,:])
			i=i+1

def writeXYZ_labelled(filename,xyz_data,labels,delimiter=' ',returns=None):
	i = 0
	with open(filename, 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=delimiter,
			quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i in range(np.size(xyz_data,0)):
			if returns is None:
				spamwriter.writerow(np.hstack((xyz_data[i,:],labels[i])))
			else:
				spamwriter.writerow(np.hstack((xyz_data[i, :], returns[i], labels[i])))
			i=i+1



def readFromBin(filename,names,formats,names_wanted=['x','y','z']):
	
	binType = np.dtype( dict(names=names, formats=formats) )
	data = np.fromfile(filename, binType)

	return np.array([data[n] for n in names_wanted]).T	


def readFromPly(filename,readOffset=True):
	plydata = PlyData.read(filename)
	offset = np.array([float(plydata.comments[0]), float(plydata.comments[1]), float(plydata.comments[2])])
	vertex = []
	for i in range(plydata.elements[0].data.shape[0]):
		vertex.append(list(plydata.elements[0].data[i]))
	vertex = np.array(vertex)

	if readOffset:
		return vertex,offset
	else:
		return vertex

def readFromOff(filename,delimiter=' '):
	data = []
	i=0
	numPoints=1e4

	with open(filename, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
		for row in spamreader:
			if i==1:
				numPoints = int(row[0])
			elif (i>1) & (i<(numPoints+2)):
				data.append(row)
			i = i+1

	data = np.array(data)
	return data.astype(float)

def readFromLas(filename,convert_colours=False, fields = ['x','y','z','Red','Green','Blue']):
	inFile = Lasfile(filename, mode='r')
	pcd = np.zeros(( inFile.x.shape[0] , len(fields) ))
	for i in range(len(fields)):
		pcd[:,i] = getattr(inFile, fields[i])
	#pcd = np.hstack((inFile.x[:, np.newaxis], inFile.y[:, np.newaxis], inFile.z[:, np.newaxis],
	#                     inFile.Red[:, np.newaxis], inFile.Green[:, np.newaxis], inFile.Blue[:, np.newaxis]))
	if convert_colours is True:
		pcd[:,3:]*=(256.0/65535.0) # do this to rgb values from 8-bit

	return pcd


# make sure directory finishes with /
def readFromDirectory(directory,type='bin',delimiter=',',x=3,y=2,z=5,label=None,returns=None,names = [],formats=[],names_wanted=['x','y','z'],output_fileList=False,readOffset=True):
	filename_list = glob.glob( ( directory + '*.%s'%(type) ) )
	xyz_list = []
	for filename in filename_list:
		if type=='bin':
			xyz_list.append(readFromBin(filename,names,formats,names_wanted=names_wanted))
		elif (type=='csv')|(type=='asc'):
			xyz_list.append(XYZreadFromCSV(filename,delimiter=delimiter,x=x,y=y,z=z,label=label,returns=returns))
		elif type=='ply':
			xyz_list.append(readFromPly(filename,readOffset=readOffset)) # dont read offset
		elif type=='off':
			xyz_list.append(readFromOff(filename,delimiter=delimiter))
		else:
			print('incorrect type')
	if output_fileList is False:
		return xyz_list
	else:
		return xyz_list,filename_list



# save pointcloud to ply file
# make sure filename has .ply extension
def writePly(filename,vertex,offset=[0,0,0,0]):
	f = open(filename, "w")
	f.write('ply\n')
	f.write('format ascii 1.0\n')
	f.write('comment %.3f\n'%(offset[0]))
	f.write('comment %.3f\n'%(offset[1]))
	f.write('comment %.3f\n'%(offset[2]))
	f.write('element vertex %d\n'%(vertex.shape[0]))
	f.write('property float x\n')
	f.write('property float y\n')
	f.write('property float z\n')
	f.write('end_header\n')
	for i in range(vertex.shape[0]):
		f.write('%.4f %.4f %.4f\n'%(vertex[i,0],vertex[i,1],vertex[i,2]))
	f.close()

# save pointcloud with labels to ply file (labels is a list)
def writePly_labelled(filename,vertex,labels,offset=[0,0,0,0]):
	vertex_withLabels=np.hstack([vertex,(np.array(labels)).reshape(-1,1)])
	f = open(filename, "w")
	f.write('ply\n')
	f.write('format ascii 1.0\n')
	f.write('comment %.3f\n'%(offset[0]))
	f.write('comment %.3f\n'%(offset[1]))
	f.write('comment %.3f\n'%(offset[2]))
	f.write('element vertex %d\n'%(vertex_withLabels.shape[0]))
	f.write('property float x\n')
	f.write('property float y\n')
	f.write('property float z\n')
	f.write('property float scalar\n')  # this is the necessary extra field
	f.write('end_header\n')
	for i in range(vertex_withLabels.shape[0]):
		f.write('%.4f %.4f %.4f %.4f\n'%(vertex_withLabels[i,0],vertex_withLabels[i,1],vertex_withLabels[i,2],vertex_withLabels[i,3]))
	f.close()


def WriteOG(og, filename=None, res=1, offset=[0,0,0], ogOffset=np.array([0,0,0]), returnPcd=False, labels=False):
	classes = np.unique(og)[1:]
	gridSize = np.shape(og)
	for i in classes:
		pc = np.array(np.nonzero(og==i),dtype=np.float)
		#if ogOffset is not None:
		pc -= (np.array(gridSize)[:,np.newaxis])/2 # was conditioned on ogOffset not none
		nPoints = np.shape(pc)[1]
		if classes.size>1:
			if i == classes[0]:
				if isinstance(res, (list)):
					vertex = np.transpose(np.vstack((pc * (np.tile(res, (nPoints, 1))).T, np.array([i] * nPoints))))
				else:
					vertex = np.transpose( np.vstack( ( pc*res , np.array( [i]*nPoints ) ) ) )
			else:
				if isinstance(res, (list)):
					vertex = np.vstack((vertex, np.transpose(np.vstack((pc * (np.tile(res, (nPoints, 1))).T, np.array([i] * nPoints))))))
				else:
					vertex = np.vstack((vertex, np.transpose(np.vstack((pc * res, np.array([i] * nPoints))))))
		else:
			if isinstance(res, (list)):
				vertex = np.transpose(pc * (np.tile(res, (nPoints, 1))).T )
			else:
				vertex = np.transpose(pc) * res
			# only one labelled class, but isnt binary (still want label column)
			if labels is True:
				vertex = np.hstack((vertex,np.zeros(np.shape(vertex)[0])[:,np.newaxis]))

	if returnPcd is False:
		if np.size( np.unique(og) ) > 1: # if pcd not empty
			if vertex.shape[1]==3:
				writePly(filename, vertex+ogOffset, offset)
			elif vertex.shape[1]==4:
				writePly_labelled(filename, vertex[:,:3]+ogOffset, offset, vertex[:,3])
		else:
			print("pcd empty")
	else:
		vertex[:, :3] = vertex[:, :3] + ogOffset
		return vertex



def WriteOG_scalar(og, filename=None, res=1, offset=[0,0,0], ogOffset=np.array([0,0,0]), returnPcd=False):
	# og is an occupancy grid with scalar values
	# only outputs points with returns > 0
	pc = np.array(np.nonzero(og > 0))
	nPoints = np.shape(pc)[1]
	vertex = np.transpose(pc * (np.tile(res, (nPoints, 1))).T)
	if returnPcd is False:
		writePly_labelled(filename, vertex+ogOffset, offset, og[og > 0])
	else:
		vertex[:, :3] = vertex[:, :3] + ogOffset
		return vertex


# def WriteOG(og, filename, res=1, offset=[0,0,0], ogOffset=None):
# 	classes = np.unique(og)[1:]
# 	gridSize = np.shape(og)
# 	for i in classes:
# 		pc = np.nonzero(og==i)
# 		if classes.size>1:
# 			if i == classes[0]:
# 				if isinstance(res, (list)):
# 					vertex = np.transpose(np.vstack((np.array(pc) * (np.tile(res, (len(pc[0]), 1))).T, np.array([i] * len(pc[0])))))
# 				else:
# 					vertex = np.transpose( np.vstack( ( np.array(pc)*res , np.array( [i]*len( pc[0] ) ) ) ) )
# 			else:
# 				if isinstance(res, (list)):
# 					vertex = np.vstack((vertex, np.transpose(np.vstack((np.array(pc) * (np.tile(res, (len(pc[0]), 1))).T, np.array([i] * len(pc[0])))))))
# 				else:
# 					vertex = np.vstack((vertex, np.transpose(np.vstack((np.array(pc) * res, np.array([i] * len(pc[0])))))))
# 		else:
# 			if isinstance(res, (list)):
# 				vertex = np.transpose(np.array(pc) * (np.tile(res, (len(pc[0]), 1))).T )
# 			else:
# 				vertex = np.transpose(np.array(pc)) * res
# 	if vertex.shape[1]==3:
# 		writePly(filename, vertex, offset)
# 	elif vertex.shape[1]==4:
# 		writePly_labelled(filename, vertex[:,:3], offset, vertex[:,3])


def og2xyz(og, og_mask, res=1, ogOffset=np.array([0,0,0]) ):
	# uses og_mask to decide which voxels are pts. Returns x,y,z and scaler value of voxel for each point

	gridSize = np.shape(og)

	pc = np.array(np.nonzero(og_mask == 1),dtype=np.float)
	# if ogOffset is not None:
	pc -= (np.array(gridSize)[:, np.newaxis]) / 2  # was conditioned on ogOffset not none
	nPoints = np.shape(pc)[1]


	if isinstance(res, (list)):
		vertex = np.transpose(np.vstack((pc * (np.tile(res, (nPoints, 1))).T, og[og_mask==1] )))
	else:
		vertex = np.transpose(np.vstack((pc * res, og[og_mask==1] )))

	vertex[:, :3] = vertex[:, :3] + ogOffset
	return vertex



def XYZreadFromCSV_general(filename,delimiter=',',x=3,y=4,z=5,other=None):
	data = []

	with open(filename, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
		for row in spamreader:
			data.append(row)

	data = np.array(data)
	if other is None:
		return data[:, [x, y, z]].astype(float)
	else:
		return data[:,[x,y,z] + other].astype(float)


def writeLAS(filename,vertex,labels=None,offset=[0,0,0],returns=None):
	'''

	:param filename: str. path/filename.las
	:param vertex: float np.array.  Nx3 of points
	:param labels: float np.array. N labels
	:param offset: list of floats e.g. [0,0,0]
	:return:
	'''
	if returns is not None:
		if len(returns.shape)>1:
			returns = returns[:,0]

	if labels is not None:
		if len(labels.shape)>1:
			labels = labels[:,0]

	vertex+=offset
	hdr = lasHeader.Header()
	outfile = Lasfile(filename, mode="w", header=hdr)
	allx = vertex[:,0]
	ally = vertex[:,1]
	allz = vertex[:,2]
	xmin = np.floor(np.min(allx))
	ymin = np.floor(np.min(ally))
	zmin = np.floor(np.min(allz))
	outfile.header.offset = [xmin,ymin,zmin]
	outfile.header.scale = [0.001,0.001,0.001]
	outfile.x = allx
	outfile.y = ally
	outfile.z = allz
	if returns is not None:
		outfile.intensity = returns
	if labels is not None:
		outfile.pt_src_id = labels
	outfile.close()

