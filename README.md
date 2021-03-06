# README medium-keypoint-autocal
This repository contains research and comparisons into three keypoint matching algorithms (SIFT, ORB, and SuperGlue) to evaluate which has the best performance both in terms of speed and accuracy compared to ground truth. 

## How to Run
run_keypoint_autocal.py is where the various keypoint matching algorithms are run depending on user input. To run it, in the medium-keypoint-autocal directory run:
```
python run_keypoint_autocal.py --max_keypoints # --datasets dataset1 --modes mode1 mode2 
```
Where the max_keypoints are the maximum matches the algorithms will look for (at least 100), the datasets are the available datasets (kitti or add your own!), and the modes are the keypoint matching algorithms (orb, sift, and/or superglue). Additional flags include ``--resize`` to resize or not resize the images, ``--do_eval`` to see performance visuals, ``--visualize`` to see the disparity maps and matches for each algorithm, and ``--match_path_exists`` to avoid running SuperGlue unnecessarily. You should only use the ``--match_path_exists`` flag if you have already run SuperGlue and want to find the SuperGlue keypoints **with the same data and without changing any other information** (same max_keypoints, same size, etc), as this flag will simply reload the last used matching keypoints.

So, to run all matching methods on the KITTI dataset and perform visualizations and analysis, you would run:
```
python run_keypoint_autocal.py --max_keypoints 500 --datasets kitti --modes sift orb superglue --do_eval --visualize
```


Before running the script with SuperGlue as a mode, please see the below section for how to install it.

## Using SuperGlue
In order to use the SuperGlue code and compare SIFT and ORB to SuperGlue, there are a few extra steps necessary. \
First, go to https://github.com/magicleap/SuperGluePretrainedNetwork. While you can clone the whole repository, the only things you'll need are the models folder and match_pairs.py. Download those two items and make sure they are both at the same level as results and data in our repository. So, once you're done, your directory should look like this: 
```
\medium-keypoint-autocal 
    data 
    keypoint_utils.py 
    LICENSE
    match_pairs.py  
    models 
    README.md 
    results 
    run_keypoint_autocal.py 
    visualize_errors.py 
```
Also, in models/utils.py, you should comment out lines 444-447, which draw lines from the keypoints in the left image to the corresponding matches in the right image. For our purposes, we just want to see the keypoints without the lines. \
The lines you should comment out are:
```
fig.lines = [matplotlib.lines.Line2D(
    (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
    transform=fig.transFigure, c=color[i], linewidth=lw)
                for i in range(len(kpts0))]
```


## Helpers
### keypoint_utils.py
This includes helper functions often used in run_keypoint_autocal.py. It includes methods for visualization, rectification, disparity map retrieval, and getting or changing extrinsic information.

### visualize_errors.py
This file contains methods for visualizing the performance of the keypoint matching algorithms for pose recovery when comparing the recovered poses to the ground truth values. It also plots the amount of time required by each matcher to compare the speed of all the approaches.


## Using the KITTI Dataset
Included in our repository are the first 6 frames from drive 15 of the KITTI Road dataset. We downloaded the unsynced+unrectified data from KITTI, used camera 2 for the left images and camera 3 for the right images, and then undistorted the images before further processing. All images should be stored in data/imgs with the following format: ``left_###.png`` or ``right_###.png`` (as seen in the repository). As more images are added, in data/pairs, more files should be added called ``kitti_pairs_###.txt`` with the number corresponding to the image number. Examples of what those files should hold can be found in the existing ``kitti_pairs_###.txt`` files. There should be the same amount of pair text files as the number of left/right image pairs.

## Citations
* SuperGlue Citation: \
Sarlin, Paul-Edouard, et al. “SuperGlue: Learning Feature Matching with Graph Neural Networks.” ArXiv.org, CVPR, 28 Mar. 2020, arxiv.org/abs/1911.11763.

* KITTI Citation: \
Geiger, Andreas, et al. “Vision Meets Robotics: The KITTI Dataset.” The International Journal of Robotics Research, vol. 32, no. 11, 2013, pp. 1231–1237., doi:10.1177/0278364913491297.

