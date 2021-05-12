import numpy as np
import cv2
from skimage.measure import block_reduce

# import skimage
# try:
#     import torch
#     import torch.nn.functional as F
#     use_torch = True
# except ImportError:
#     use_torch = False
#     print("You'd better pip install pytorch")
# def gauss_down4(img):
#     down_1 = cv2.pyrDown(img)
#     down_2 = cv2.pyrDown(down_1)
#     return down_2
def box_down2(input, name=None):
    return block_reduce(input, block_size=(2,2), func=np.mean)

def gauss_down4(input, name=None):
    img = np.zeros((input.shape[0] // 4,input.shape[1] // 4,input.shape[2]))
    for i in range(input.shape[2]):
        temp = cv2.GaussianBlur(input[:,:,i], (5,5), -1)
        img[:,:,i] = block_reduce(temp, block_size=(4,4), func=np.mean)
    return img

def gauss_7x7(input, name=None):
    img = cv2.GaussianBlur(input, 7, -1)
    return img

def diff(im1, im2, name=None):
    return im1 - im2
  
T_SIZE=32           # Size of a tile in the bayer mosaiced image
T_SIZE_2=16         # Half of T_SIZE and the size of a tile throughout the alignment pyramid

MIN_OFFSET=-168     # Min total alignment (based on three levels and downsampling by 4)
MAX_OFFSET=126      # Max total alignment. Differs from MIN_OFFSET because total search range is 8 for better vectorization

DOWNSAMPLE_RATE=4   # Rate at which layers of the alignment pyramid are downsampled relative to each other

# prev_tile -- Returns an index to the nearest tile in the previous level of the pyramid.
def prev_tile(t): return (t - 1) / DOWNSAMPLE_RATE

# tile_0 -- Returns the upper (for y input) or left (for x input) tile that an image index touches.
def tile_0(e): return e / T_SIZE_2 - 1

# tile_1 -- Returns the lower (for y input) or right (for x input) tile that an image index touches.
def tile_1(e): return e / T_SIZE_2

# idx_0 -- Returns the inner index into the upper (for y input) or left (for x input) tile that an image index touches.
def idx_0(e): return e % T_SIZE_2  + T_SIZE_2

# idx_1 -- Returns the inner index into the lower (for y input) or right (for x input) tile that an image index touches.
def idx_1(e): return e % T_SIZE_2

# idx_im -- Returns the image index given a tile and the inner index into the tile.
def idx_im(t, i): return t * T_SIZE_2 + i

# idx_layer -- Returns the image index given a tile and the inner index into the tile.
def idx_layer(t, i): return t * T_SIZE_2 / 2 + i


# align -- Aligns multiple raw RGGB frames of a scene in T_SIZE x T_SIZE tiles which overlap
# by T_SIZE_2 in each dimension. align(imgs)(tile_x, tile_y, n) is a point representing the x and y offset
# for a tile in layer n that most closely matches that tile in the reference (relative to the reference tile's location)