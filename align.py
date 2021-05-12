import numpy as np
import cv2
from utils import gauss_down4
from merge import merge_temporal
import time
t_size = 16 # 块大小
t_size_2 = 8 # stride
downsample_rate = 4 # 下采样的系数
search_range = 4 # 搜索半径
# 对齐函数，输入一组图像帧，返回图像相对于第一帧（参考帧）的块偏移
def align(img):
    layer_0 = gauss_down4(img) # 生成上一层gaussican金字塔，长宽缩放4倍
    layer_1 = gauss_down4(layer_0) # 生成上一层gaussican金字塔，长宽缩放4倍
    layer1_shape = layer_1.shape 
    # 生成初始偏移估计
    init_alignment = np.zeros((layer1_shape[0] // (t_size_2 * downsample_rate) + 1,layer1_shape[1] // (t_size_2 * downsample_rate) + 1,2,layer1_shape[2]),dtype=np.int16)
    # 根据上一层的偏移估计计算当前层的偏移估计
    alignment_1 = align_pyramid_layer(layer_1,init_alignment)
    alignment_0 = align_pyramid_layer(layer_0,alignment_1)
    alignment = align_pyramid_layer(img,alignment_0)
    return alignment
    # cv2.imwrite('D:/test/alignment.png',alignment[:,:,1])
    # print(alignment[:,:,1].mean())
    

# 金字塔偏移函数，以上一层金字塔块偏移为基础，计算当前层的块偏移
def align_pyramid_layer(layer,prev_alignment):
    layer = cv2.copyMakeBorder(layer,0,t_size,0,t_size,cv2.BORDER_REFLECT) # 图像边缘补充
    align_shape = prev_alignment.shape
    layer_shape = layer.shape
    # 根据上一层偏移，生成当前层的初始估计偏移
    prev_align_offset = prev_alignment.repeat(downsample_rate,axis = 0) 
    prev_align_offset = prev_align_offset.repeat(downsample_rate,axis = 1)
    prev_align_offset *= downsample_rate
    for n in range(1,align_shape[3]):
        for i in range(0,layer_shape[0] - t_size,t_size_2):
            for j in range(0,layer_shape[1] - t_size,t_size_2):
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
    img0 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(0)+".png",0).astype('float64') 
    # img0 = cv2.imread("D:/RGBW_denoising/data/move_data/png/test_1lux_01_merge.png",0).astype('float64') / 2
    img1 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(1)+".png",0).astype('float64') 
    img2 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+".png",0).astype('float64') 
    img3 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png",0).astype('float64')
    # img4 = cv2.imread("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png",0).astype('float64')
    #
    gauss0 = cv2.GaussianBlur(img0, (5,5), -1)   
    gauss1 = cv2.GaussianBlur(img1, (5,5), -1)
    gauss2 = cv2.GaussianBlur(img2, (5,5), -1)
    gauss3 = cv2.GaussianBlur(img3, (5,5), -1)
    imgs = [img0,img1,img2,img3]
    gauss = [gauss0,gauss1,gauss2,gauss3]
    # gauss4 = cv2.GaussianBlur(img4, (5,5), -1)
    
    frame = np.concatenate((img1[:,:,np.newaxis],img0[:,:,np.newaxis]),axis=2) * 4
    gauss_frame = np.concatenate((gauss1[:,:,np.newaxis],gauss0[:,:,np.newaxis]),axis=2) * 4
    time_start=time.time()
    alignment = align(gauss_frame)
    time_end=time.time()
    print('aligement time cost',time_end-time_start,'s')
    output = merge_temporal(gauss_frame,alignment,frame)
    time_end_2=time.time()
    print('merge time cost',time_end_2-time_end,'s')
    # cv2.imwrite("D:/RGBW_denoising/data/move_data/png/test_1lux_10_merge.png",output / 4 * 2)
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
    


    






