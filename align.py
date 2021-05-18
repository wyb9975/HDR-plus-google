import numpy as np
import cv2
from merge import merge_temporal
import time
from utils import *
from motion import motion_mask
t_size = 16 # 块大小
t_size_2 = 8 # stride
downsample_rate = 4 # 下采样的系数
search_range = 4 # 搜索半径
# 对齐函数，输入一组图像帧，返回图像相对于第一帧（参考帧）的块偏移
def align(img,mask):
    layer_0 = gauss_down4(img) # 生成上一层gaussican金字塔，长宽缩放4倍
    layer_1 = gauss_down4(layer_0) # 生成上一层gaussican金字塔，长宽缩放4倍
    layer1_shape = layer_1.shape 
    # 生成初始偏移估计
    init_alignment = np.zeros((layer1_shape[0] // (t_size_2 * downsample_rate) + 1,layer1_shape[1] // (t_size_2 * downsample_rate) + 1,2,layer1_shape[2]),dtype=np.int16)
    # 根据上一层的偏移估计计算当前层的偏移估计
    time_start=time.time()
    alignment_1 = align_pyramid_layer(layer_1,init_alignment)
    time_end=time.time()
    print('layer 1 aligement time cost',time_end-time_start,'s')
    time_start=time.time()
    alignment_0 = align_pyramid_layer(layer_0,alignment_1)
    time_end=time.time()
    print('layer 2 aligement time cost',time_end-time_start,'s')
    time_start=time.time()
    alignment = align_pyramid_layer(img,alignment_0)
    time_end=time.time()
    print('layer 3 aligement time cost',time_end-time_start,'s')
    return alignment
    # cv2.imwrite('D:/test/alignment.png',alignment[:,:,1])
    # print(alignment[:,:,1].mean())
    

# 金字塔偏移函数，以上一层金字塔块偏移为基础，计算当前层的块偏移
def align_pyramid_layer(layer,prev_alignment,mask=np.array([0])):
    layer = cv2.copyMakeBorder(layer,0,t_size,0,t_size,cv2.BORDER_REFLECT) # 图像边缘补充
    if len(mask.shape) == 2:
        mask = cv2.copyMakeBorder(mask,0,t_size,0,t_size,cv2.BORDER_REFLECT)
    align_shape = prev_alignment.shape
    layer_shape = layer.shape
    # 根据上一层偏移，生成当前层的初始估计偏移
    prev_align_offset = prev_alignment.repeat(downsample_rate,axis = 0) 
    prev_align_offset = prev_align_offset.repeat(downsample_rate,axis = 1)
    prev_align_offset *= downsample_rate
    for n in range(1,align_shape[3]):
        for i in range(0,layer_shape[0] - t_size,t_size_2):
            for j in range(0,layer_shape[1] - t_size,t_size_2):
                if len(mask.shape) == 1 or np.mean(mask[i:i + t_size,j:j+t_size]) > 0.1:
                    offset = prev_align_offset[i // t_size_2,j // t_size_2,:,n] # 当前位置的初始估计偏移
                    prev_align_offset[i // t_size_2,j // t_size_2,:,n]  = patch_match(layer,i + offset[0],j + offset[1],i,j,n) 
    return prev_align_offset


    
# 在估计坐标周围进行窗口搜索，搜索最佳的匹配位置，返回相对于参考帧块坐标的偏移
# layer为当前金字塔层
# x,y为第n帧这一块的估计位置，ref_x，ref_y为参考帧的块位置
def patch_match(layer,x,y,ref_x,ref_y,n):
    # match_position = Point()
    layer_shape = layer.shape
    match_position = [0,0]
    distance = 1000000 # 最小距离
    ref_img = layer[ref_x:ref_x + t_size,ref_y:ref_y + t_size,0] #参考帧图像块
    # i，j为搜索范围[-4,3],
    for i in range(-search_range,search_range):
        if x + i < 0 or x + i >= layer_shape[0] - t_size:
            continue
        for j in range(-search_range,search_range):
            if  y + j < 0 or y + j >= layer_shape[1] - t_size:
                continue
            cur_img = layer[x + i:x + i + t_size,y + j:y + j + t_size,n]
            diff = ref_img-cur_img
            temp = np.linalg.norm(diff.reshape(-1),ord=1)
            if distance > temp:
                match_position[0] = x + i - ref_x
                match_position[1] = y + j - ref_y
                distance = temp
    # print(match_position)
    return match_position

if __name__ == '__main__':

    print('hello')
    width = 2560
    height = 1440
    # img0 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(0)+".png",0).astype('float64') 
    # # img0 = cv2.imread("D:/RGBW_denoising/data/move_data/png/test_1lux_01_merge.png",0).astype('float64') / 2
    # img1 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(1)+".png",0).astype('float64') 
    # img2 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+".png",0).astype('float64') 
    # img3 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png",0).astype('float64')
    # img4 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png",0).astype('float64')
    gain = 48
    wp = 1023
    p = sample_VST(gain = gain)
    input = 'D:/RGBW_denoising/data/move_data2/rgbw_20f.raw'
    raw = np.fromfile(input, dtype=np.uint16)
    num = 2
    group = 1
    begin = 2
    w_denoi = np.zeros((2560*1440*10),dtype = np.uint16)
    for j in range(group):
        frame = np.zeros((height,width,num),dtype='float32')
        gauss_frame = np.zeros_like(frame)
        for i in range(num):
            frame[:,:,num - 1 - i] = inter_w(raw[height*width*(begin +i+j):height*width*(begin + i + j + 1)].reshape((height,width)).astype('float32') // 4)
            gauss_frame[:,:,num - 1 - i] = cv2.GaussianBlur(frame[:,:,num - 1 - i], (5,5), -1) 

        # img0 = cv2.imread("D:/RGBW_denoising/data/move_data/png/test_1lux_01_merge.png",0).astype('float64') / 2
        # img1 = inter_w(raw[height*width*1:height*width*2].reshape((height,width)).astype('float32') // 4) 
        # #
        # gauss0 = cv2.GaussianBlur(img0, (5,5), -1)   
        # gauss1 = cv2.GaussianBlur(img1, (5,5), -1)
        # gauss2 = cv2.GaussianBlur(img2, (5,5), -1)
        # gauss3 = cv2.GaussianBlur(img3, (5,5), -1)
        # imgs = [img0,img1,img2,img3]
        # gauss = [gauss0,gauss1,gauss2,gauss3]
        # gauss4 = cv2.GaussianBlur(img4, (5,5), -1)
        
        # frame = np.concatenate((img1[:,:,np.newaxis],img0[:,:,np.newaxis]),axis=2) 
        # gauss_frame = np.concatenate((gauss1[:,:,np.newaxis],gauss0[:,:,np.newaxis]),axis=2)
        gauss_frame /= 1023
        time_start=time.time()
        mask = motion_mask(gauss_frame[:,:,0], gauss_frame[:,:,1], gain=gain)
        time_end=time.time()
        print('mask time cost',time_end-time_start,'s')
        print(mask.shape)
        gauss_frame *= 1023
        time_start=time.time()
        alignment = align(gauss_frame,mask)
        time_end=time.time()
        print('aligement time cost',time_end-time_start,'s')
        output = merge_temporal(gauss_frame,alignment,frame,mask)
        time_end_2=time.time()
        print('merge time cost',time_end_2-time_end,'s')
        # cv2.imwrite("D:/RGBW_denoising/data/move_data2/test/f" + str(num) + "_refence" + str(begin + num - 1) + "_fact4_max10.png",output / 4 * 2)
        output = output.astype(np.uint16).reshape((width*height))
        w_denoi[j*width*height:(j+1)*width*height] = output
    # w_denoi.tofile('D:/RGBW_denoising/data/move_data2/w_doenoi_f5_10.raw')

    # for i in range(2,3):
    #     frame = np.concatenate((imgs[i][:,:,np.newaxis],output[:,:,np.newaxis]),axis=2) * 4
    #     gauss_frame = np.concatenate((gauss[i][:,:,np.newaxis],gauss[i - 1][:,:,np.newaxis]),axis=2) * 4
    #     alignment = align(gauss_frame)
    #     output = merge_temporal(gauss_frame,alignment,frame)
    #     cv2.imwrite("D:/RGBW_denoising/data/move_data/png/test_1lux_12_merge.png",output / 4 * 2)
    # print(np.mean(alignment))
    # shape = alignment.shape
    # result = np.zeros((720,1280))
    # img = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_norm_"+ str(0)+".png",0)
    # img2 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_norm_"+ str(1)+".png",0)
    # img2 = cv2.copyMakeBorder(img2,0,t_size,0,t_size,cv2.BORDER_REFLECT)

    # for i in range(1440 // 8 ):
    #     for j in range(2560 // 8 ):
    #         img[i*8:(i+1) * 8,j*8:(j+1)*8] = img2[i*8+alignment[i][j][0]:(i+1) * 8 +alignment[i][j][0],j*8 + alignment[i][j][1]:(j+1)*8 + alignment[i][j][1]]
    


    






