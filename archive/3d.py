
import time
import os
import cv2
import exifread

import numpy as np

from sys import argv
from sklearn.neighbors import NearestNeighbors
from random import randrange as rand

# EXIF tags
EXIF_FOCAL_LENGTH = 'EXIF FocalLength'

# RANSAC / fundamental matrix
MAX_ITERATIONS = 1000
TH_ACCEPT = 0.40
TH_PX = 3.0

K = [ 	[ 3.71413737e3, 0.0, 1.48020701e3],
		[ 0.0, 3.72821333e3, 1.17327469e3],
		[ 0.0, 0.0, 1.0]
	]

class Feature(object):
	kp = None
	des = None
	K = None
	
def computeKMatrix(shape, tags):
	''' Computes the camera's intrisic properties matrix K
		return K or None
	'''
	try:
		fEXIF = str(tags[EXIF_FOCAL_LENGTH])
	except KeyError:
		return None
	
	if '/' in fEXIF:
		n, d = fEXIF.split('/')
		f = float(n) / float(d)
	else:
		f = float(fEXIF)
	
	x0 = shape[0] / 2.0
	y0 = shape[1] / 2.0
	
	K = [ [ f/shape[0], 0, x0], [0, f/shape[1], y0], [0, 0, 1] ]
	
	return K
		
def pt(kp):
	''' extracts the xy-point from the keypoint
	'''
	return [ float(kp[0]), float(kp[1]) ]
	
def readAndExtract(filePath):
	''' Reads the file at filePath and extracts SIFT keypoints
		and EXIF focal length
		returns keypoints, focal length for image
	'''
	# get the focal length from the picture
	f = open(filePath, 'rb')
	tags = exifread.process_file(f, stop_tag=EXIF_FOCAL_LENGTH, details=False)
	f.close()
	
	# read image as grayscale
	gray = cv2.imread(filePath, 0)
	
	# downsample image if too large
	shp = np.shape(gray)
	scale = 1.0
	nPixels = shp[0]*shp[1]
	if nPixels > 2000000:
		scale = float(2000000) / nPixels
		gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
	
	# get intrinsic matrix K
	K = computeKMatrix(shp, tags)
	
	# get SIFT features as a vector
	sift = cv2.SIFT()
	keyPoints, descriptors = sift.detectAndCompute(gray, None)
	
	return keyPoints, descriptors, K

def getFeaturesFromDir(dir):
	''' Extracts SIFT features / focal lengths from every file in dir
	''' 
	features = []
	
	for fileName in os.listdir(dir):
		filePath = os.path.join(dir, fileName)
		if os.path.isfile(filePath) and filePath[-3:] != '.db':
			f = Feature()
			f.kp, f.des, f.K = readAndExtract(filePath)
			features.append(f)

	return features
	
def buildNNGraph(features):
	''' Performs nearest neighbor search using a KD tree
		returns a list of edges of the form: 
	'''
	graph = []
	for i, feat1 in enumerate(features):
		kd = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(feat1.des)
		graph.append({})
		for j in range(i+1, len(features)):
			feat2 = features[j]
			distances, nn_i = kd.kneighbors(feat2.des)
			
			for d_i, dist in enumerate(distances):
				if (dist[0] / dist[1]) < 0.8: #passes Lowe's Ratio test
					try:
						graph[i][j].append( (feat1.kp[nn_i[d_i][0]].pt, feat2.kp[d_i].pt) )
					except KeyError:
						graph[i][j] = []
						graph[i][j].append( (feat1.kp[nn_i[d_i][0]].pt, feat2.kp[d_i].pt) )
	return graph

def yVector(yp, y):
	return [ 	
				yp[0]*y[0], 
				yp[0]*y[1], 
				yp[0], 
				yp[1]*y[0],
				yp[1]*y[1],
				yp[1],
				y[0],
				y[1],
				1
			]

def eightPoint(corrs):
	''' computes the Fundamental matrix
	'''
	Y = []
	pts1 = np.array([(0, 0)]*len(corrs))
	pts2 = np.array([(0, 0)]*len(corrs))
	for i, c in enumerate(corrs):
		c1 = c[0]
		c2 = c[1]
		Y.append(yVector( c1, c2) )
		pts1[i] = c1
		pts2[i] = c2

	u, w, vt = np.linalg.svd(Y)
	m, mask = cv2.findFundamentalMat( np.float32(pts1), np.float32(pts2) )
	
	# find the vector corresponding to smallest singular value from SVD
	s = np.reshape(vt[np.argmin(w)], (3, 3) )
	
	# return the vector as the F matrix
	return s# np.reshape(s, (3, 3) )
	
def distance(a, b):
	''' Compute the euclidean distance between two vectors
	'''
	
	return sum([ (a[i] - b[i])**2 for i in range(len(a))])**0.5

def getInliers(F, corrs):
	''' Returns all correspondences that are within th_px of computed value
	'''
	
	inliers = []
	for c in corrs:
		x = [c[0][0], c[0][1], 1]
		x_prime = [c[1][1], c[1][1], 1]
		x_est = np.dot(F, x)
		d = np.dot(x_prime, x_est)
		if abs(d) <= TH_PX:
			inliers.append(c)
	
	return inliers

def confidence(F, corrs):
	''' Helper function for ransack()
		computes the percent of correspondences that are within th_px of the computed value
	'''
	
	# count the number of correspondences that match the criteria
	n = 0.0
	for c in corrs:
		x = [c[0][0], c[0][1], 1]
		x_prime = [c[1][0], c[1][1], 1]
		x_est = np.dot(F, x)
		d = np.dot(x_prime, x_est)
		
		if abs(d) <= TH_PX:
			n += 1.0
	
	# return percentage
	return n / len(corrs)

def computeFMatrix(corrs):
	''' Compute the fundamental matrix form matching correspondences
	''' 
	
	n = len(corrs)
	if n < 8:
		return None
		
	Fbest = []
	cbest = 0.0
	
	for i in range(MAX_ITERATIONS):
		sample = [ corrs[rand(n)] for i in range(8) ]
		
		F = eightPoint(sample)
		
		c = confidence(F, corrs)
		if c > TH_ACCEPT:
			Fbest = F
			break
		elif c > cbest:
			Fbest = F
			cbest = c
	print cbest
	I = getInliers(Fbest, corrs)
	F = eightPoint(I)
	return F
	
def computeFEMatrices(graph, features):
	''' Get the fundamental matrices for all matching image pairs
	'''
	Es = []
	for i, img1 in enumerate(graph):
		if features[i].K == None:
			print 'None'
			continue
		else:
			Es.append({})
			for key in img1.keys():
				if features[key].K == None:
					print 'None'
					continue
				else:
					print 'computing F'
					F = computeFMatrix(img1[key])
					# compute the Essential matrix
					E = F
					Es[i][key] = E
						
	return Fs
	
def computePMatrices(Es):

	for i in range(len(Es)):
		
		for key in Es[i].keys():
			E = Es[i][key]
			P = E
			
	return P

if __name__ == '__main__':
	if len(argv) != 2:
		print 'Usage:'
		print argv[0], 'ImageDirectory'
		
	else:
		print 'Extracting SIFT Features...'
		features = getFeaturesFromDir(argv[1])
		print 'Building NN Graph...'
		graph = buildNNGraph(features)
		
		#skeleton graph
		print 'Computing Fundamental Essential Matrices'
		Es = computeFEMatrices(graph, features)#RANSAC for fundamental / essential matrix
		#compute Projection Matrix
		Ps = computePMatrices(Es)
		#track generation
		
	
	