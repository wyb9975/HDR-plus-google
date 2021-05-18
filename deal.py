import cv2
import numpy as np
from numpy import cumsum
from numpy import zeros
from numpy import tile
def pngEnlarge(input,output,enlarge = 2):
    img = cv2.imread(input,0) * enlarge
    cv2.imwrite(output,img)
def boxfilter(imSrc, r):

    # BOXFILTER O(1) time box filtering using cumulative sum
    # - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    # - Running time independent of r;
    # - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
    # - But much faster.

    hei, wid = imSrc.shape
    imDst = zeros((hei, wid), dtype=np.float32)
    # print(imDst.dtype)

    # cumulative sum over Y axis
    imCum = cumsum(imSrc, axis=0)
    # difference over Y axis
    imDst[0:r + 1, :] = imCum[r:2 * r + 1, :]
    imDst[r + 1:hei - r, :] = imCum[2 * r + 1:hei, :] - imCum[0: hei - 2 * r - 1, :]
    imDst[hei - r:hei, :] = tile(imCum[hei - 1, :], (r, 1)) - imCum[hei - 2 * r - 1:hei - r - 1, :]

    # cumulative sum over X axis
    imCum = cumsum(imDst, axis=1)
    # difference over Y axis
    imDst[:, 0:r+1] = imCum[:, r:2 * r + 1]
    imDst[:, r + 1:wid - r] = imCum[:, 2 * r + 1:wid] - imCum[:, 0:wid - 2 * r - 1]
    imDst[:, wid - r:wid] = tile(imCum[:, wid - 1:wid], (1, r)) - imCum[:, wid - 2 * r - 1:wid - r - 1]
    return imDst
def demosaick_W(raw):
    row, col = raw.shape
    W = raw.copy()

    # handle the boundary
    row_idx = [1, row-2, 2]
    col_idx = [2, col-1, 2]
    idx_row = np.arange(1, row - 2, 2)
    idx_col = np.arange(2, col - 1, 2)
    W[row - 1, 0] = (W[row - 2, 0] + W[row - 1, 1]) / 2  # corner
    W[row_idx[0]:row_idx[1]:row_idx[2], 0] = (W[row_idx[0]-1:row_idx[1]-1:row_idx[2], 0]
                                              + W[row_idx[0]+1:row_idx[1]+1:row_idx[2], 0]) / 2  # left edge
    W[row - 1, col_idx[0]:col_idx[1]:col_idx[2]] = (W[row - 1, col_idx[0]-1:col_idx[1]-1:col_idx[2]]
                                                    + W[row - 1, col_idx[0]+1:col_idx[1]+1:col_idx[2]]) / 2  # down edge

    #  for the rest using the horizontal and vertical directions
    rr = idx_row.shape[0]
    cc = idx_col.shape[0]
    W_hat = np.zeros((rr, cc, 2))  # estimated value of W
    G = np.zeros((rr, cc, 2))  # abs of gradient
    G2 = np.zeros((rr, cc, 2))  # second-order gradient

    #  horizontal
    G2[:, :, 0] = ((W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                    + W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]] 
                    - 2 * W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]]) / 2
                   + (W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                      + W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]] 
                      - 2 * W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]]) / 2
                   ) / 2
    W_hat[:, :, 0] = (W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                      + W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]]) / 2 - G2[:, :, 0]
    G[:, :, 0] = np.abs(W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                        - W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]]) \
                 + np.abs(G2[:, :, 0] * 2)

    #  vertical
    G2[:, :, 1] = ((W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                    + W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]] 
                    - 2 * W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]+1:col_idx[1]+1:col_idx[2]]) / 2
                   + (W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]] 
                      + W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]] 
                      - 2 * W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]-1:col_idx[1]-1:col_idx[2]]) / 2
                   ) / 2
    W_hat[:, :, 1] = (W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]] 
                      + W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]]) / 2 - G2[:, :, 1]
    G[:, :, 1] = np.abs(W[row_idx[0]+1:row_idx[1]+1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]] 
                        - W[row_idx[0]-1:row_idx[1]-1:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]]) \
                 + np.abs(G2[:, :, 1] * 2)

    #  get the final estimated values
    r = 1  # kernel size [2*r+1,2*r+1]
    for ii in range(0, 2):
        G[:, :, ii] = boxfilter(G[:, :, ii], r) / ((2 * r + 1) ** 2)  # smooth the weights
    zero_thresh = 1e-10  # in the 16bit domain
    weight = 1 / (G + zero_thresh)
    weight /= np.sum(weight, axis=2)[:, :, np.newaxis]
    W[row_idx[0]:row_idx[1]:row_idx[2], col_idx[0]:col_idx[1]:col_idx[2]] = np.sum(W_hat * weight, axis=2)

    return W
def crosstalk_W(raw):
    # cross-talk reduction for W channel
    row, col = raw.shape
    W = raw.copy()
    row_idx = [2, row-1, 2]
    col_idx = [1, col-2, 2]
    
    # boundary
    W[0,col-1] = ( W[0,col-1]+W[0,col-2]+W[1,col-1] )/3
    W[0,col_idx[0]:col_idx[1]:col_idx[2]] = ( W[0,col_idx[0]-1:col_idx[1]-1:col_idx[2]] + W[0,col_idx[0]+1:col_idx[1]+1:col_idx[2]] + W[0,col_idx[0]:col_idx[1]:col_idx[2]] + W[1,col_idx[0]:col_idx[1]:col_idx[2]] )/4
    W[row_idx[0]:row_idx[1]:row_idx[2],col-1] = ( W[row_idx[0]-1:row_idx[1]-1:row_idx[2],col-1] + W[row_idx[0]+1:row_idx[1]+1:row_idx[2],col-1] + W[row_idx[0]:row_idx[1]:row_idx[2],col-2] + W[row_idx[0]:row_idx[1]:row_idx[2],col-1] )/4
    
    # cross-talk
    rr = (row-1 - 2 - 1) // 2 + 1
    cc = (col-2 - 1 - 1) // 2 + 1
    W_hat = np.zeros((rr,cc,2))# estimated value of W
    G = np.zeros((rr,cc,2))#  abs. of gradient
    # horizontal
    G[:,:,0] = abs( W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]+1:col_idx[1]+1:col_idx[2]]+W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]-1:col_idx[1]-1:col_idx[2]]-2*W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]] )
    W_hat[:,:,0] = ( W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]] + W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]+1:col_idx[1]+1:col_idx[2]]+W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]-1:col_idx[1]-1:col_idx[2]] ) / 3
    # vertical
    G[:,:,1] = abs( W[row_idx[0]+1:row_idx[1]+1:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]]+W[row_idx[0]-1:row_idx[1]-1:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]]-2*W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]] )
    W_hat[:,:,1] = (W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]] + W[row_idx[0]+1:row_idx[1]+1:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]]+W[row_idx[0]-1:row_idx[1]-1:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]]) / 3
    # get the final estimated values
    r = 1# kernel size [2*r+1,2*r+1]
    for ii in range(0, 2):
        G[:,:,ii] = boxfilter( G[:,:,ii], r ) # smooth the weights
    zero_thresh = 1# in the 16bit domain
    weight = 1 / (G+zero_thresh)
    weight /= np.sum(weight, axis=2)[:, :, np.newaxis]
    W[row_idx[0]:row_idx[1]:row_idx[2],col_idx[0]:col_idx[1]:col_idx[2]] = np.sum(W_hat*weight, axis=2)
    return W
# pngEnlarge("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+".png","D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(2)+"_2x.png")
# pngEnlarge("D:/RGBW_denoising/data/move_data/png/test_1lux_merge_limit.png","D:/RGBW_denoising/data/move_data/png/test_1lux_merge_limit_2x.png")
# pngEnlarge("D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+".png","D:/RGBW_denoising/data/move_data/png/rgbw_1lux_"+ str(3)+"_2x.png")

