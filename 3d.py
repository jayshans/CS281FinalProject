import time
import os
import cv2

import numpy as np

from sys import argv
from sklearn.neighbors import NearestNeighbors
from random import randrange as rand
from random import sample

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# EXIF tags
EXIF_FOCAL_LENGTH = 'EXIF FocalLength'

# RANSAC / fundamental matrix
MAX_ITERATIONS = 1000
SHARED_PTS_LIMIT = 1000

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
	
	# img = cv2.drawKeypoints(gray, keyPoints)
	# cv2.imshow(filePath, img)
	# cv2.waitKey(0);
	
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
				if (dist[0] < 0.8*dist[1]): #passes Lowe's Ratio test
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
	return m
	
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
	Ibest = []
	for i in range(MAX_ITERATIONS):
		sample = [ corrs[rand(n)] for i in range(8) ]
		
		F = eightPoint(sample)
		I = getInliers(F, corrs)
		c = float(len(I)) / len(corrs)
		if c > TH_ACCEPT:
			Fbest = F
			cbest = c
			Ibest = I
			break
		elif c > cbest:
			Fbest = F
			cbest = c
			Ibest = I
			

	F = eightPoint(Ibest)
	return F, Ibest
	
def computeFEMatrices(graph, features):
	''' Get the fundamental matrices for all matching image pairs
	'''
	Es = []
	for i, img1 in enumerate(graph):
		Es.append({})
		for key in img1.keys():
			#print imageNames[i], 'vs', imageNames[key]
			#F, graph[i][key] = computeFMatrix(img1[key])
			F, I = computeFMatrix(img1[key])
			# compute the Essential matrix
			E = np.dot( Kt, np.dot( F, K) )
			Es[i][key] = E

	return Es, graph
	
	
	
def computePMatrices(Es, graph):

	Ps = []
	for i in range(len(Es)):
		Ps.append({})
		for key in Es[i].keys():
			E = Es[i][key]
			U, s, V = np.linalg.svd(E, full_matrices=True)
			P = chooseP(U, s, V, graph[i][key][0][0], graph[i][key][0][1])
			Ps[i][key] = P
			
	return Ps
	
def correctP(point1, point2, R, T):
    
		
	b = np.dot(R[0, :] - point2[0]*R[2, :], T) / np.dot(R[0, :] - point2[0]*R[2, :], np.mat([point2[0], point2[1], 1]).T)
	z = b[0, 0]
	Z1 = np.array([point1[0] * z, point1[1] * z, z])
	Z2 = np.dot(R.T, Z1) - np.dot(R.T, T)

	if Z1[2] < 0 or Z2[0, 2] < 0:
		return False
	else:
		return True
	

def chooseP(U, s, V, point1, point2):

	W = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
	R = np.mat(U) * W * np.mat(V)
	T=np.mat(U[:, 2]).T
	
	#4 Possible solutions for P' and test them to ensure points are projected in front of the cameras
	if not correctP(point1, point2, R, T):

		T = - np.mat(U[:, 2]).T
		
		if not correctP(point1, point2, R, T):

			R = U.dot(W.T).dot(V)
			T = np.mat(U[:, 2]).T

			if not correctP(point1, point2, R, T):

				T = - np.mat(U[:, 2]).T

	P=R
	P = np.dot(K, np.append( P, T, axis = 1 ))

	return P

def plotReconstruction(XYZ):
	X = [x[0] for x in XYZ]
	Y = [y[1] for y in XYZ]
	Z = [z[2] for z in XYZ]
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	#plt.axis('off')
	ax.scatter(X, Y, Z, c='black', s=5, depthshade=False)
	ax.get_yaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.set_aspect('equal')
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

def getSharedPoints(numberOfImages, keepingTrack):
	
	dataStructure = [[] for i in range(numberOfImages*3)]
	
	for track in keepingTrack:
		for i in range(numberOfImages*3):
			dataStructure[i].append(None)
		for points in track:
			image = points[2]
			dataStructure[image*3][-1] = points[0]
			dataStructure[image*3+1][-1] = points[1]
			dataStructure[image*3+2][-1] = 1
	
	largestAmountSharedPoints = 0
	largestSharedPointsSet = []
	
	for j in range(len(SHARED_PTS_LIMIT)):
		random_images = sample(range(numberOfImages), 3)
		random_images_indices = [random_images[0]*3, random_images[0]*3+1, random_images[0]*3+2, 
								random_images[1]*3, random_images[1]*3+1, random_images[1]*3+2,
								random_images[2]*3, random_images[2]*3+1, random_images[2]*3+2]
								
		sharedPoints = np.array(dataStructure[random_images_indices])
		counter = 0
		while(counter < len(sharedPoints[0])):
			if sharedPoints[0,counter] == None or sharedPoints[3,counter] == None or sharedPoints[6,counter] == None:
				sharedPoints = np.delete(sharedPoints, counter, 1)
			else:
				counter += 1
		if len(sharedPoints[0]) > largestAmountSharedPoints:
			largestAmountSharedPoints = len(sharedPoints[0])
			largestSharedPointsSet = sharedPoints
	#print sharedPoints
	#print len(sharedPoints), len(sharedPoints[0]), len(sharedPoints[1]), len(sharedPoints[2]) 
	
	return largestAmountSharedPoints
	
def testpoint(Ps, graph):

	I = np.mat([[ 1, 0, 0, 0],[ 0, 1, 0, 0], [ 0, 0, 1, 0]])
	points3D=[]
	for i in range(len(Ps)):
		points3D.append({})
		for key in Ps[i].keys():
			pt1 = np.array([corr[0] for corr in graph[i][key]]).T
			pt2 = np.array([corr[1] for corr in graph[i][key]]).T
			
			pnts = cv2.triangulatePoints(I, Ps[i][key], pt1, pt2)
			points3D[i][key] = pnts.T

	return points3D
	
def getZ(m, P, S):

	Z = np.copy(m)
	for i in range(0, len(m), 3):
	
		P_i = np.mat(P[i:i+3])
		
		mZ_i = P_i * S
		m_i = m[i:i+3, :]
		
		z_iEst = np.divide(mZ_i, m_i)
		z_i = z_iEst[2]
		
		Z[i] = z_i
		Z[i+1] = z_i
		Z[i+2] = z_i

	return np.mat(Z)
	
def recon(m):

	r, c = np.shape(m)
	U, d, Vt = np.linalg.svd(m, full_matrices = False)
	print 'm', np.shape(m)
	#initial estimates
	P = np.mat(np.dot( U[:, :4], np.diag(d[:4])))
	S = np.mat(Vt[:4, :])
	
	for someNumber in range(1000):
		if d[4] < 0.01:
			print 'S', np.shape(S)
			return S
		else:
			# estimate projective scaling\
			Z = getZ(m, P, S)
			W = np.multiply(m, Z)
			
			U, d, Vt = np.linalg.svd(W, full_matrices=False)
			
			P = np.mat(np.dot( U[:, :4], np.diag(d[:4])))
			S = np.mat(Vt[:4, :])
			
	return None

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
				
		sharedPoints = getSharedPoints(len(graph), keepingTrack)
		
		pnts = recon(sharedPoints)
		pnts = np.array(np.transpose(pnts))
		print len(pnts)
		if type(pnts) != type(None):
			plotReconstruction(pnts)
		exit()
		
		
		#Skeleton Graph
		print 'Computing Fundamental Essential Matrices'
		#RANSAC for fundamental / essential matrix
		Es, graph = computeFEMatrices(graph, features)
		
		#compute Projection Matrix
		print 'Computing Projection Matrices'
		Ps = computePMatrices(Es, graph)

		points3D = testpoint(Ps, graph)
		print np.shape(points3D[0][1])
		pts = []
		for key in points3D[0].keys():
			for p in points3D[0][key]:
				pts.append(p)
		
		plotReconstruction(points3D[0][1])
	'''#plotting test
	a = [[3,3,3],[2,1,5],[6,2,1],[0,2,5]]
	plotReconstruction(a)
	'''

