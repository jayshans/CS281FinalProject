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


# RANSAC / fundamental matrix
MAX_ITERATIONS = 1000
SHARED_PTS_LIMIT = 100

TH_ACCEPT = 0.40
TH_PX = 2.0

K = np.mat([ 
	[ 3.71413737e3, 0.0, 1.48020701e3],
	[ 0.0, 3.72821333e3, 1.17327469e3],
	[ 0.0, 0.0, 1.0]
	])
Kt = K.T

imageNames = []
	
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
	T=np.mat(U[:, 2])
	
	#4 Possible solutions for P' and test them to ensure points are projected in front of the cameras
	if not correctP(point1, point2, R, T):

		T = - np.mat(U[:, 2])
		
		if not correctP(point1, point2, R, T):

			R = U.dot(W.T).dot(V)
			T = np.mat(U[:, 2])

			if not correctP(point1, point2, R, T):

				T = - np.mat(U[:, 2])

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
	ax.scatter(X, Y, Z, c='black', s=1, depthshade=False)
	ax.get_yaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	#ax.set_aspect('equal')
	plt.show()
	
def get3DPoints(kp1, kp2, des1, des2):

	kd = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(des1)
	distances, indexes = kd.kneighbors(des2)
	
	pnts1 = []
	pnts2 = []
	
	for i, dist in enumerate(distances):
		if dist[0] < 0.8*dist[1]:
			pnts1.append( kp1[ indexes[i][0] ].pt )
			pnts2.append( kp2[ i ].pt )

	F, mask = cv2.findFundamentalMat( np.float32(pnts1), np.float32(pnts2) )
	E = K.T * np.mat(F) * K
	
	p1 = []
	p1Ind = []
	p2 = []
	p2Ind = []
	for i, e in enumerate(mask):
		if e == 1:
			p1.append(pnts1[i])
			p1Ind.append(i)
			p2.append(pnts2[i])
			p2Ind.append(i)
	
	pnts1 = np.array(p1).T
	pnts2 = np.array(p2).T
	
	U, s, V = np.linalg.svd(E, full_matrices=True)
	P = chooseP(U, s, V, pnts1[0], pnts2[0])
	
	I = np.mat([[ 1, 0, 0, 0],[ 0, 1, 0, 0], [ 0, 0, 1, 0]])
	
	pnts3d = cv2.triangulatePoints(K*I, P, pnts1, pnts2)
	
	for i, pnt in enumerate(pnts3d):
		pnts3d[i] = np.divide(pnt, pnts3d[3])
	
	return pnts3d, pnts1, pnts2

if __name__ == '__main__':
	if len(argv) != 4:
		print 'Usage:'
		print argv[0], 'image1', 'image2', 'image3'
	else:
		img1 = cv2.imread(argv[1])
		img2 = cv2.imread(argv[2])
		img3 = cv2.imread(argv[3])
		
		gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
		
		sift = cv2.SIFT()
		kp1, des1 = sift.detectAndCompute(gray1, None)
		kp2, des2 = sift.detectAndCompute(gray2, None)
		kp3, des3 = sift.detectAndCompute(gray3, None)
		
		I = np.mat([[ 1, 0, 0, 0],[ 0, 1, 0, 0], [ 0, 0, 1, 0]])
		
		pnts3d1, placeHolder, placeHolder = get3DPoints(kp1, kp2, des1, des2)
		pnts3d2, placeHolder, placeHolder = get3DPoints(kp1, kp3, des1, des3)
		
		p1 = pnts3d1[:3].T
		p2 = pnts3d2[:3].T
		
		plotReconstruction(p1)
		plotReconstruction(p2)
		
		if len(p1) > len(p2):
			minLength = len(p2)
		else:
			minLength = len(p1)
		
		#print p2
		retval, out, inliers = cv2.estimateAffine3D(p2[:minLength], p1[:minLength])
		print out
		print inliers
		
		out = np.vstack((out, np.array([0,0,0,1])))
		
		
		p2 = p2.T
		p2 = np.vstack((p2, np.ones(len(p2[0]))))
		p2 = np.dot(p2.T, out)
		#pnts3d2 = np.dot(A, pnts3d2)
		
		#p = np.append( p1.T, p2.T, axis = 0 )
		
		'''pointsForImage3 = np.array([[i.pt[0], i.pt[1]] for i in kp3])
		
		if len(p1) > len(pointsForImage3):
			minLength = len(pointsForImage3)
		else:
			minLength = len(p1)
		
		retval, rvec, tvec = cv2.solvePnP(p1[:minLength], pointsForImage3[:minLength], K, None)

		rvec = cv2.Rodrigues(rvec)[0]
		print rvec
		print tvec
		
		P = np.append(rvec, tvec, axis = 1)
		print P
		
		P = np.vstack((P, np.array([0,0,0,1])))
		print P
		
		pnts3d2, placeHolder, placeHolder = get3DPoints(kp1, kp3, des1, des3)
		
		p1 = p1.T
		p1 = np.vstack((p1, np.ones(len(p1[0])))).T
		
		p2 = pnts3d2[:3] 
		
		p2 = np.vstack((p2, np.ones(len(p2[0])))).T
		
		p2 = np.dot(p2, P)
		'''
		print np.shape(p1)
		print np.shape(p2)
		
		print p1
		print p2[:,:3]
		p = np.append( p1, p2[:,:3], axis = 0 )
		
		plotReconstruction(p)
		
		
		
		
		