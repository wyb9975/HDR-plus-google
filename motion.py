
import numpy as np
import cv2
from utils import *
def motion_mask(img, img_prev, gain=48, depth=4, scale=2):
    diffs = []
    WeightMap = [None] * depth
    p = sample_VST(gain = gain)
    sigRead = p['sigRead'] / 1023
    print(sigRead)
    for i in range(depth): 
        img = gauss_down(img, scale)
        img_prev = gauss_down(img_prev, scale)
        diff = np.abs(img - img_prev)
        WeightMap[i] = np.zeros_like(diff)
        WeightMap[i][diff>sigRead*6/scale/(np.sqrt(scale)**i)] = 1
        diffs.append(norm(diff))
        # img = gauss_down(img)
        # img_prev = gauss_down(img_prev)
    # PyrShow(WeightMap, 'map')
    # PyrShow(diffs, 'diff', False)
    weight = np.ones(depth)
    mask = WeightMap[-1]
    for i in range(depth-1):
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
        mask = mask * weight[depth-i-1] + WeightMap[depth-i-2] * weight[depth-i-2]
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    mask = np.clip((mask-1),0,(np.sum(weight) - 1)) / (np.sum(weight) - 1)
    # cv2.imwrite("D:/RGBW_denoising/data/move_data/png/test_1lux_10_merge.png",mask *255)
    # ImgShow(mask, 'mask')
    return mask