import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import yaml
from pathlib import Path
from models.utils import plot_matches, plot_image_pair


def read(pathname: str):
    ''' 
    Read intrinsics from yaml file 
    d = intrinsics dictionary
    K = intrinsic camera matrix
    distCoeffs = distortion coefficients
    '''
    fid = open(pathname)
    d = yaml.load(fid, Loader=yaml.FullLoader)
    fid.close()
    
    # Intrinsic camera matrix
    K = np.array([[d['fx'], 0,       d['cx']], 
                  [0,       d['fy'], d['cy']], 
                  [0 ,      0,       1]])
    
    # Distortion coefficients
    distCoeffs = np.array([d['k1'], d['k2'], d['p1'], d['p2'],
                           d['k3'], d['k4'], d['k5'], d['k6']])

    return d, K, distCoeffs


def rectify(left: np.ndarray, right: np.ndarray, K1: np.ndarray, distCoeffs1: np.ndarray, 
    K2: np.ndarray, distCoeffs2: np.ndarray, R: np.ndarray, T: np.ndarray):
    ''' 
    Rectification of images.
    '''
    size = (left.shape[1], left.shape[0])  # e.g., (1280,720)
    R1, R2, P1, P2, _Q, _validPixROI1, _validPixROI2 = \
        cv.stereoRectify(K1, distCoeffs1, K2, distCoeffs2, size, R, T)

    map1x, map1y = cv.initUndistortRectifyMap(K1, distCoeffs1, R1, P1, size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(K2, distCoeffs2, R2, P2, size, cv.CV_32FC1)

    left_rectified = cv.remap(left, map1x, map1y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
    right_rectified = cv.remap(right, map2x, map2y, cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
    
    return left_rectified, right_rectified


def get_disparity(left: np.ndarray, right: np.ndarray, maxDisparity=256):
    '''
    Gets disparity map.
    '''
    # requires grayscale images
    if np.ndim(left) == 3:
        left  = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        right = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    wsize = 5  # SAD window size
    alg = cv.StereoSGBM_create(minDisparity = 0,
                                numDisparities = maxDisparity, 
                                blockSize = wsize,
                                P1 = wsize*wsize*4,
                                P2 = wsize*wsize*32,
                                disp12MaxDiff = 1,
                                preFilterCap = 63,
                                uniquenessRatio = 10,
                                speckleWindowSize = 100,
                                speckleRange = 32,
                                mode = cv.STEREO_SGBM_MODE_HH)

    # Compute Disparity and Divide by DISP_SCALE=16
    disparity = alg.compute(left, right).astype(np.float32)/16.0

    return disparity


def parse_matches(path: str):
    '''
    Helper for SuperGlue. Parses the files in which SuperGlue puts the 
    matching pairs to get separate corresponding lists of matching points.
    '''
    npz = np.load(path)
    left_keypts = npz["keypoints0"]
    right_keypts = npz["keypoints1"]
    matches = npz["matches"]
    confidences = npz["match_confidence"]

    left_matched = []
    right_matched = []
    new_confidences = []

    for i in range(len(matches)):
        if matches[i] != -1 and confidences[i] > .5:
            left_matched.append(left_keypts[i])
            right_matched.append(right_keypts[matches[i]])
            new_confidences.append(confidences[i])

    return np.array(left_matched), np.array(right_matched), np.array(new_confidences)


def rotm2euler(R: np.ndarray):
    '''
    Converts R to euler angles.
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def euler2rotm(theta):   
    '''
    Converts euler angles to R.
    ''' 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def recover_pose(points1: list, points2: list, K1: np.ndarray, K2: np.ndarray):
    '''
    Recovers the pose from the matching keypoints and intrinsics.
    '''
    # Find the Fundamental Matrix
    F, mask = cv.findFundamentalMat(points1, points2, cv.FM_RANSAC, 0.1, 0.999)
    
    # Find the Essential Matrix
    E = K2.T @ F @ K1

    # Recover pose
    _retval, R, t, mask = cv.recoverPose(E, points1, points2, K2)

    T = t.T[0]  # unit vector
    
    return R, T, F, E


def visualize_match_distribution(left_imgs: list, right_imgs: list, matches: dict, modes: list, 
    dataset: str, frames: list):
    ''' 
    Shows matches found by all matching algorithms across the image pair.
    '''
    colors = ["red", "green", "blue"]

    for i in range(len(frames)):
        plot_image_pair([left_imgs[i], right_imgs[i]])
        for j in range(len(modes)):
            left_matched = matches[frames[i]][j][0]
            right_matched = matches[frames[i]][j][1]
            plot_matches(left_matched, right_matched, [colors[j]] * len(left_matched))

        plt.legend(modes, loc=1)
        path = "results/matches/" + dataset + "_overlay/"
        match_dir = Path(path)
        match_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(path + dataset + "_" + frames[i] + "_all_matchers.png")