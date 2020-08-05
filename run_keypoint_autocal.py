import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import yaml
from keypoint_utils import *
from models.utils import scale_intrinsics, read_image, process_resize, make_matching_plot
from visualize_errors import *
import time
from os import listdir, system
import argparse
from pathlib import Path
import torch

# ground truth rotation and translation for KITTI
kitti_R_gt = np.degrees(np.array([-0.0010622,  -0.01971669, -0.02230844]))
kitti_T_gt = np.array([-0.53267121,  0.00526146, -0.00782809])

device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'


def get_SuperGlue_keypoints(input_pair: str, out_dir: str, npz_name: str, max_keypoints: int, visualize: bool, 
    resize: list, match_path_exists: bool):
    '''
    Gets the keypoints using SuperGlue feature matching.
    Inputs:
        - input_pair: name of text file for left/right pair
        - out_dir: directory where matches are stored after SuperGlue is run
        - npz_name: name of match file
        - max_keypoints: maximum number of matching keypoints that should be found
        - visualize: indicates whether or not to visualize the matches
        - resize: dimension to resize image
        - match_path_exists: indicates not to perform SuperGlue matching if the matches are already stored
    Outputs:
        - mkpts1: matched keypoints for left image
        - mkpts2: matched keypoints for right image
    '''
    if not match_path_exists:
        script = './match_pairs.py --input_dir {} --input_pairs {} --output_dir {} \
                --superglue {} --max_keypoints {} --nms_radius 3 --resize_float'
        if visualize:
            script += ' --viz'
        if len(resize) == 2:
            script += ' --resize {} {}'
            system(script.format('data/', input_pair, out_dir, 'outdoor', max_keypoints, resize[0], resize[1]))
        elif len(resize) == 1:
            script += ' --resize {}'
            system(script.format('data/', input_pair, out_dir, 'outdoor', max_keypoints, resize[0]))

    mkpts1, mkpts2, confidences = parse_matches("data/matches/" + npz_name + "_matches.npz")

    best_confidences = np.argsort(confidences)[-100:]
    return mkpts1[best_confidences], mkpts2[best_confidences]


def get_SIFT_keypoints(img1: np.ndarray, img2: np.ndarray, max_keypoints: int):
    '''
    Gets the keypoints using SIFT feature matching.
    Inputs:
        - img1: left image
        - img2: right image
        - max_keypoints: maximum number of matching keypoints that should be found
    Outputs:
        - mkpts1: matched keypoints for left image
        - mkpts2: matched keypoints for right image
    '''
    sift = cv.xfeatures2d.SIFT_create(max_keypoints)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    des1 = np.asarray(des1, np.float32)
    des2 = np.asarray(des2, np.float32)

    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    matches_full_img = bf.match(des1, des2)
    matches_full_img = sorted(matches_full_img, key = lambda x:x.distance)
    matches_full_img = matches_full_img[:100]
   
    mkpts1 = np.array([kp1[m.queryIdx].pt for m in matches_full_img]) # queryIdx indexes des1 or kp1
    mkpts2 = np.array([kp2[m.trainIdx].pt for m in matches_full_img]) # trainIdx indexes

    return mkpts1, mkpts2


def get_ORB_keypoints(img1: np.ndarray, img2: np.ndarray, max_keypoints: int):
    '''
    Gets the keypoints using ORB feature matching.
    Inputs:
        - img1: left image
        - img2: right image
        - max_keypoints: maximum number of matching keypoints that should be found
    Outputs:
        - mkpts1: matched keypoints for left image
        - mkpts2: matched keypoints for right image
    '''
    orb = cv.ORB_create(max_keypoints)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    matches_full_img = bf.match(np.asarray(des1), np.asarray(des2))
    matches_full_img = sorted(matches_full_img, key = lambda x:x.distance)
    matches_full_img = matches_full_img[:100]

    mkpts1 = np.array([kp1[m.queryIdx].pt for m in matches_full_img]) # queryIdx indexes des1 or kp1
    mkpts2 = np.array([kp2[m.trainIdx].pt for m in matches_full_img]) # trainIdx indexes des2 or kp2
    
    return mkpts1, mkpts2 

    
def get_keypoints(img1: np.ndarray, img2: np.ndarray, max_keypoints: int, num_img: str, visualize: bool, 
    resize: list, match_path_exists: bool, dataset: str, mode: str):
    '''
    Retrieves pose from the keypoints of the images based on the given keypoint matching algorithm. 
    Inputs:
        - img1: left image (undistorted)
        - img2: right image (undistorted)
        - max_keypoints: maximum number of keypoints matching tool should consider
        - num_img: frame number
        - visualize: indicates whether visualizations should be done and saved
        - resize: dimensions at which images should be resized 
        - match_path_exists: indicates for SuperGlue if there are saved matches or if it should redo the matches
        - mode: keypoint matching algorithm in use
        - dataset: current dataset being evaluated
    Outputs:
        - R: recovered rotation matrix
        - T: recovered translation vector
        - mkpts1: matched keypoints in left image
        - mkpts2: matched keypoints in right image
    '''

    left, _inp, left_scale = read_image(img1, device, resize, 0, False)
    right, _inp, right_scale = read_image(img2, device, resize, 0, False)
    left = left.astype('uint8')
    right = right.astype('uint8')

    i1, K1, distCoeffs1 = read("data/intrinsics/" + dataset + "_left.yaml")
    i2, K2, distCoeffs2 = read("data/intrinsics/" + dataset + "_right.yaml")

    K1 = scale_intrinsics(K1, left_scale)
    K2 = scale_intrinsics(K2, right_scale)

    if mode == "superglue":
        input_pair = "data/pairs/kitti_pairs_" + num_img + ".txt"
        npz_name = "left_" + num_img + "_right_" + num_img
        out_dir = "data/matches/"
        mkpts1, mkpts2 = get_SuperGlue_keypoints(input_pair, out_dir, npz_name, max_keypoints, visualize, 
            resize, match_path_exists)
    elif mode == "sift":
        mkpts1, mkpts2 = get_SIFT_keypoints(left, right, max_keypoints)
    elif mode == "orb":
        mkpts1, mkpts2 = get_ORB_keypoints(left, right, max_keypoints)

    R, T, F, _E = recover_pose(mkpts1, mkpts2, K1, K2)

    left_rectified, right_rectified = rectify(left, right, K1, distCoeffs1, K2, distCoeffs2, R, kitti_T_gt) 

    if visualize:        
        text = [mode, "Best 100 of " + str(max_keypoints) + " keypoints"]
        colors = np.array(['red'] * len(mkpts1))
        res_path = str("results/matches/" + mode + "/")
        match_dir = Path(res_path)
        match_dir.mkdir(parents=True, exist_ok=True)
        path = res_path + dataset + "_" + mode + "_matches_" + num_img + ".png"
        make_matching_plot(left, right, mkpts1, mkpts2, mkpts1, mkpts2,
                        colors, text, path, show_keypoints=False,
                        fast_viz=False, opencv_display=False,
                        opencv_title='matches', small_text=[])


        save_disp_path = "results/disparity/" + mode + "/"
        disp_dir = Path(save_disp_path)
        disp_dir.mkdir(parents=True, exist_ok=True)
        disp = get_disparity(left_rectified, right_rectified, maxDisparity=128)
        plt.imsave(save_disp_path + dataset + "_" + mode + "_disp_" + num_img + ".png", disp, cmap="jet")

    return R, T, mkpts1, mkpts2


def parse_args():
    '''
    Parses arguments from the command line.

    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--modes", type=str, nargs="+", default=["superglue"], help="keypoint matching algorithms (orb, sift, or superglue)"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["kitti"], help="dataset(s) being evaluated"
    )
    parser.add_argument(
        "--resize", type=int, nargs="+", default=[1600],
        help="one or two numbers for determining new dimensions; can input -1 to not resize"
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=500, help="maximum keypoints matchers can find, must be greater than 100"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="perform error evaluations"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="perform and save disparity and match visualizations"
    )
    parser.add_argument(
        "--match_path_exists", action="store_true", help="indicate if superglue matches already exist"
    )

    opt = parser.parse_args()

    if len(opt.modes) == 0 or len(opt.datasets) == 0:
        raise ValueError("Must provide at least one string value for --modes and --datasets")
    elif opt.max_keypoints < 100:
        raise ValueError("--max_keypoints must be at least 100")
    
    return opt.modes, opt.datasets, opt.do_eval, opt.visualize, opt.max_keypoints, opt.resize, opt.match_path_exists



if __name__ == "__main__":
    modes, datasets, do_eval, visualize, max_keypoints, resize, match_path_exists = parse_args()
    
    frame_matches_per_mode = {}

    for dataset in datasets:
        filenames = listdir("data/imgs")
        img_nums = np.arange(0, int(len(filenames)/2)).astype(str)
        img_nums = [img_nums[i].zfill(3) for i in range(len(img_nums))]
        for num in img_nums:
            frame_matches_per_mode[num] = []
        
        R_ests = []
        times_per_mode = []
        for mode in modes:
            times = []
            Rs = []
            total_pitch = 0
            total_yaw = 0
            total_roll = 0

            for i in range(len(img_nums)):
                img1_path = "data/imgs/left_" + img_nums[i] + ".png"
                img2_path = "data/imgs/right_" + img_nums[i] + ".png"

                start = time.time()
                R, T, mkpts1, mkpts2 = get_keypoints(img1_path, img2_path, max_keypoints, img_nums[i], visualize, 
                    resize, match_path_exists, dataset, mode)
                end = time.time()
                times.append(end - start)

                R_degrees = np.degrees(rotm2euler(R))
                Rs.append(R_degrees)

                total_pitch += R_degrees[0]
                total_yaw += R_degrees[1]
                total_roll += R_degrees[2]

                frame_matches_per_mode[img_nums[i]].append([mkpts1, mkpts2])
                print("finished pair " + str(i))

            R_ests.append(Rs)
            times_per_mode.append(times)
            
            if do_eval:
                total_pitch /= len(img_nums)
                total_yaw /= len(img_nums)
                total_roll /= len(img_nums)

                R_avg = euler2rotm(np.radians(np.array([total_pitch, total_yaw, total_roll])))

                left = cv.imread("data/imgs/left_000.png", 0)
                right = cv.imread("data/imgs/right_000.png", 0)
                i1, K1, distCoeffs1 = read("data/intrinsics/" + dataset + "_left.yaml")
                i2, K2, distCoeffs2 = read("data/intrinsics/" + dataset + "_right.yaml")

                left_rectified, right_rectified = rectify(left, right, K1, distCoeffs1, K2, distCoeffs2, R_avg, kitti_T_gt) 

                disp = get_disparity(left_rectified, right_rectified, maxDisparity=128)
                save_disp_path = "results/disparity/" + mode + "/"
                disp_dir = Path(save_disp_path)
                disp_dir.mkdir(parents=True, exist_ok=True)
                plt.imsave(save_disp_path + dataset + "_" + mode + "_avg_pose_disp.png", disp, cmap="jet")


        if do_eval:
            show_errors(R_ests, kitti_R_gt, dataset, modes)

            left_imgs = []
            right_imgs = []
            for i in range(len(img_nums)):
                img1_path = "data/imgs/left_" + img_nums[i] + ".png"
                img2_path = "data/imgs/right_" + img_nums[i] + ".png"

                left, _inp, _left_scale = read_image(img1_path, device, resize, 0, False)
                right, _inp, _right_scale = read_image(img2_path, device, resize, 0, False)
                
                left_imgs.append(left)
                right_imgs.append(right)
                
            visualize_match_distribution(left_imgs, right_imgs, frame_matches_per_mode, modes, dataset, img_nums)
           
            if not visualize and not match_path_exists:
                plot_times(dataset, times_per_mode, modes)
        
            frame_matches_per_mode.clear()
        print("finished " + dataset)
