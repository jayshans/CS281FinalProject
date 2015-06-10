import time
import os
import cv2

import numpy as np

from sys import argv
from sklearn.neighbors import NearestNeighbors
from random import randrange as rand

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# EXIF tags
EXIF_FOCAL_LENGTH = 'EXIF FocalLength'

# RANSAC / fundamental matrix
MAX_ITERATIONS = 1000
TH_ACCEPT = 0.40
TH_PX = 2.0

K = [ 
	[ 3.71413737e3, 0.0, 1.48020701e3],
	[ 0.0, 3.72821333e3, 1.17327469e3],
	[ 0.0, 0.0, 1.0]
	]
Kt = np.transpose(K)

imageNames = []

class Feature(object):
	kp = None
	des = None

		
def pt(kp):
	''' extracts the xy-point from the keypoint
	'''
	return [ float(kp[0]), float(kp[1]) ]
	
def readAndExtract(filePath):
	''' Reads the file at filePath and extracts SIFT keypoints
		and EXIF focal length
		returns keypoints, focal length for image
	'''	
	# read image as grayscale
	gray = cv2.imread(filePath, 0)
	
	# downsample image if too large
	shp = np.shape(gray)
	scale = 1.0
	nPixels = shp[0]*shp[1]
	if nPixels > 2000000:
		scale = float(2000000) / nPixels
		gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
	
	# get SIFT features as a vector
	sift = cv2.SIFT()
	keyPoints, descriptors = sift.detectAndCompute(gray, None)
	
	return keyPoints, descriptors

def getFeaturesFromDir(dir):
	''' Extracts SIFT features / focal lengths from every file in dir
	''' 
	features = []
	
	for fileName in os.listdir(dir):
		imageNames.append(fileName)
		filePath = os.path.join(dir, fileName)
		if os.path.isfile(filePath) and filePath[-3:] != '.db':
			f = Feature()
			f.kp, f.des = readAndExtract(filePath)
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
	return s
	
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
	
	print np.shape(inliers)
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
			cbest = c
			break
		elif c > cbest:
			Fbest = F
			cbest = c
	print cbest
	print confidence(Fbest, corrs)
	I = getInliers(Fbest, corrs)
	print np.shape(I)
	F = eightPoint(I)
	return F
	
def computeFEMatrices(graph, features):
	''' Get the fundamental matrices for all matching image pairs
	'''
	Es = []
	for i, img1 in enumerate(graph):
		Es.append({})
		for key in img1.keys():
			print imageNames[i], 'vs', imageNames[key]
			F = computeFMatrix(img1[key])
			# compute the Essential matrix
			E = np.dot( Kt, np.dot( F, K) )
			Es[i][key] = E
					
	return Es
	
	
	
def computePMatrices(Es, graph):

	for i in range(len(Es)):
		
		for key in Es[i].keys():
			E = Es[i][key]
			[U, s, V ]= np.linalg.svd(E, full_matrices=True)
			Q=chooseP(U, s, V, np.mat([ [2], [2], [1] ]))
                        P=Q
			
	return P
	
def correctP(point1, point2, R, T):
    
    for first, second in Tip(point1, point2):
		
        z = np.dot(R[0, :] - second[0]*R[2, :], T) / np.dot(R[0, :] - second[0]*R[2, :], second)
        Z1 = np.array([first[0] * z, second[0] * z, z])
        Z2 = np.dot(R.T, Z1) - np.dot(R.T, T)

        if Z1[2] < 0 or Z2[2] < 0:
            return False

    return True
	

def chooseP(U, s, V, apoint):

	R = np.mat(U) * np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) * np.mat(V)
	T=U[:, 2]
	
	#4 Possible solutions for P' and test them to ensure points are projected in front of the cameras
	if not correctP(point1, point2, R, T):

		T = - U[:, 2]
		
		if not correctP(point1, point2, R, T):

			R = U.dot(W.T).dot(Vt)
			T = U[:, 2]

			if not correctP(point1, point2, R, T):

				T = - U[:, 2]

	P=R
	P[:, 3]=T
	

	return P

def plotReconstruction(XYZ):
	X = [x[0] for x in XYZ]
	Y = [x[1] for x in XYZ]
	Z = [x[2] for x in XYZ]
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	plt.axis('off')
	ax.scatter(X, Y, Z, c='black', depthshade=False)
	ax.get_yaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_Tticklabels([])
	ax.set_aspect(3, None, 'C')
	plt.show()
	
def buildTracks(graph):
	keepingTrack = []

	for image in range(len(graph)):
		for match in graph[image].keys():
			for points in graph[image][match]:
				x1 = points[0][0]
				y1 = points[0][1]
				x2 = points[1][0]
				y2 = points[1][1]
				
				point1 = (x1, y1, image)
				point2 = (x2, y2, match)
				
				for i in range(len(keepingTrack)):
					if keepingTrack[i][-1] == point1:
						keepingTrack[i].append(point2)
						continue
				
				keepingTrack.append([point1,point2])		
	
	return keepingTrack

def findMaxTrackLength(keepingTrack):
	tracksMaxLength = 0
	for track in keepingTrack:
		if len(track) > tracksMaxLength:
			tracksMaxLength = len(track)
	return tracksMaxLength
	
if __name__ == '__main__':
	if len(argv) != 2:
		print 'Usage:'
		print argv[0], 'ImageDirectory'
		
	else:
		#Preprocessing
		print 'Extracting SIFT Features...'
		features = getFeaturesFromDir(argv[1])
		print 'Building NN Graph...'
		graph = buildNNGraph(features)
		
		#Track Generation
		print 'Tracking...'
		keepingTrack = buildTracks(graph)
		print 'buildTracks Length:', len(keepingTrack)
		tracksMaxLength = findMaxTrackLength(keepingTrack)
		print 'Longest buildTracks Length:', tracksMaxLength
		
		#Skeleton Graph
		print 'Computing Fundamental Essential Matrices'
		#RANSAC for fundamental / essential matrix
		Es = computeFEMatrices(graph, features)
		
		#compute Projection Matrix
		print 'Computing Projection Matrices'
		Ps = computePMatrices(Es, graph)
		
	'''#plotting test
	a = [[3,3,3],[2,1,5],[6,2,1],[0,2,5]]
	plotReconstruction(a)
	'''
