import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import math
from pathlib import Path

colors = ["darkblue", "magenta", "cyan", "lime"]


def show_errors(ests: list, gt: np.ndarray, dataset: str, modes: list):
    '''
    Visualizes the errors for datasets over multiple frames.
    Inputs:
        - ests: estimates 
        - gt: ground truth 
        - dataset: dataset
        - modes: keypoint matching tools used for recovering pose
    '''
    graphs = ["pitch", "yaw", "roll"]
    
    fig, axes = plt.subplots(3, figsize=(18,15))

    num_imgs = np.arange(0, int(len(listdir("data/imgs"))/2))
    
    for k in range(len(graphs)):      
        mode_ests = []
        print("ground truth " + graphs[k] + " estimate: " + str(gt[k]))
        for i in range(len(modes)):  
            all_ests = []
            est_total = 0
            for j in range(len(num_imgs)):
                all_ests.append(ests[i][j][k])
                est_total += ests[i][j][k]
            
            mode_ests.append(all_ests)
        
            print(modes[i] + " " + graphs[k] + " avg error: " + str(abs((est_total/len(num_imgs)) - gt[k])))
            
        print("***********************************************************")
        
        gt_vals = [gt[k] for i in range(len(num_imgs))]
        
        for i in range(len(modes)):
            axes[k].plot(num_imgs, mode_ests[i], marker='', color=colors[i], linewidth=2, label=modes[i])
        axes[k].plot(num_imgs, gt_vals, linestyle='dashed', marker='', color=colors[-1], linewidth=2, label="ground truth")

        axes[k].set(xlabel="Image Frames", ylabel="Estimate (degrees)")
        axes[k].set_title("Keypoint matcher performance for " + graphs[k] + ", ground truth = " + str(gt[k]))

    axes[0].legend(loc=2)
    path = "results/performance_eval/"
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path + dataset + "_rotation_eval.png")


def plot_times(dataset: str, times: list, modes: list):
    '''
    Plots the amount of time each mode in use takes per frame.
    Inputs:
        - dataset: dataset 
        - time: times per frame per mode
        - modes: keypoint matching tools used for recovering pose
    '''
    fig, ax = plt.subplots()    
    num_imgs = len(times[0])
    
    for i in range(len(modes)):
        plt.plot(np.arange(num_imgs), times[i], marker='', color=colors[i], linewidth=2, label=modes[i])

    plt.xlabel("Image Frames")
    plt.ylabel("Time (seconds)")
    plt.title("Keypoint Matcher Speeds")
    plt.legend(loc=2)
    plt.tight_layout()

    path = "results/performance_eval/"
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path + dataset + "_time_eval.png")