'''This script is for calculating camera pose with respect to chessboard.

1. Specify the size and the square width of chessboard.
2. Specify the file name of camera parameters.
3. Press 'q' to quit.
'''

import cv2
import numpy as np
import yaml
import time
from scipy.spatial.transform import Rotation as rot
import util

camera = cv2.VideoCapture(0)
CHESSBOARD_CORNER_NUM_X = 9
CHESSBOARD_CORNER_NUM_Y = 6
SQUARE_WIDTH=26.24
CAMERA_PARAMETERS_INPUT_FILE = "cam1.yaml"

frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARD_CORNER_NUM_X*CHESSBOARD_CORNER_NUM_Y,3), np.float32)
for i in range( CHESSBOARD_CORNER_NUM_Y ) :
    for j in range( CHESSBOARD_CORNER_NUM_X ) :
        objp[i*CHESSBOARD_CORNER_NUM_X+j,0]=j*SQUARE_WIDTH
        objp[i*CHESSBOARD_CORNER_NUM_X+j,1]=i*SQUARE_WIDTH

axis = np.float32([[SQUARE_WIDTH,0,0], [0,SQUARE_WIDTH,0], [0,0,-SQUARE_WIDTH]]).reshape(-1,3)

# Load camera intrinsic parameters.
with open(CAMERA_PARAMETERS_INPUT_FILE) as f:
    loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), flags=cv2.CALIB_CB_FAST_CHECK)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors by PnP.
        t0 = time.time()
        ret,rvec,tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        t1 = time.time()
        r = rot.from_rotvec(rvec.T).as_euler('xyz', degrees=True)
        cv2.putText(img, 'Finding camera pose using PnP({:.3f}ms):'.format(1000*(t1-t0)), (20, int(frame_height - 210)), cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 255, 255))
        cv2.putText(img, 'Rotation(Euler angles): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(r[0][0], r[0][1], r[0][2]), (20, int(frame_height) - 170), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        cv2.putText(img, 'Translation(mm): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(tvec[0][0], tvec[1][0], tvec[2][0]), (20, int(frame_height) - 130), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        # Find the rotation and translation vectors by Homography.
        t0 = time.time()
        corners3 = cv2.undistortPoints(corners2, cameraMatrix=mtx, distCoeffs=dist, P=mtx)
        H, mask = cv2.findHomography(objp, corners3, cv2.RANSAC, 5.0)
        if H is not None:
            (R, T) = util.camera_pose_from_homography(mtx, H)
            rvec_, _ = cv2.Rodrigues(R.T)
            t1 = time.time()
            cv2.putText(img, 'Finding camera pose using homography({:.3f}ms):'.format(1000*(t1-t0)), (20, int(frame_height - 90)), cv2.FONT_HERSHEY_SIMPLEX, .5, (100, 255, 255))
            r = rot.from_rotvec(rvec_.T).as_euler('xyz', degrees=True)
            cv2.putText(img, 'Rotation(Euler angles): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(r[0][0], r[0][1], r[0][2]), (20, int(frame_height)  - 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
            cv2.putText(img, 'Translation(mm): X: {:0.2f} Y: {:0.2f} Z: {:0.2f}'.format(T[0], T[1], T[2]), (20, int(frame_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))
        imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        img = util.draw(img,corners2,imgpts)
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break;
cv2.destroyAllWindows()

