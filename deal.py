import cv2
def pngEnlarge(input,output,enlarge = 2):
    img = cv2.imread(input,0) * enlarge
    cv2.imwrite(output,img)
# pngEnlarge("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+".png","D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+"_2x.png")
# pngEnlarge("D:/RGBW_denoising/data/move_data/png/test_1lux_merge_limit.png","D:/RGBW_denoising/data/move_data/png/test_1lux_merge_limit_2x.png")
pngEnlarge("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png","D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+"_2x.png")

