import numpy as np
import cv2
from skimage.measure import block_reduce
from deal import demosaick_W
from deal import crosstalk_W
import skimage
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

def inter_w(w):
    w = crosstalk_W(w)
    w = demosaick_W(w)
    return w
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

gain2ISO_table = np.array([105,160,242,387,555,841,1273,1946,2951,4466,
    6760,10232,15488,23442,35892,54325,82224,124451,188364,285101,436515])
ISO2K_table = {'ISO':(0,4466,6760),'k':(0.0009798960625162485,0.000994945642373262,
    0.0009105033663420229),'b':(0.0025932979098808318,-0.06021553706411087,0.31690366769140255)}
    
def gain2ISO(gain):
    # {'k': 0.08315924147199373, 'b': 4.663095717798595, 'sig': 0.011518394114624558}
    idx = gain // 5
    alpha = gain % 5
    if idx >= 20: return gain2ISO_table[idx]
    ISO = ((5-alpha) * gain2ISO_table[idx] + alpha * gain2ISO_table[idx+1]) / 5
    return ISO
    
def ISO2K(ISO):
    for i in range(len(ISO2K_table['ISO'])-1):
        if ISO < ISO2K_table['ISO'][i+1]:
            k = ISO2K_table['k'][i]
            b = ISO2K_table['b'][i]
            return k * ISO + b
    k = ISO2K_table['k'][-1]
    b = ISO2K_table['b'][-1]
    return ISO * k + b

def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['Huawei'] = { # gain = [40,50]
        'Kmin':2.41104, 'Kmax':3.22505, 'lam':-0.100, 'q':1/(2**12), 'wp':4095,
        'sigRk':0.92085,  'sigRb':-2.81694,  'sigRsig':0.01856,
        'sigGsk':0.91148, 'sigGsb':0.23734, 'sigGssig':0.00214,
        'sigReadk':0.91451, 'sigReadb':0.22752, 'sigReadsig':0.00202
    }
    cam_noisy_params['NikonD850'] = {
        'Kmin':1.2, 'Kmax':2.4828, 'lam':-0.26, 'q':1/(2**14), 'wp':16383,
        'sigTLk':0.906, 'sigTLb':-0.6754,   'sigTLsig':0.035165,
        'sigRk':0.8322,  'sigRb':-2.3326,   'sigRsig':0.301333,
    }
    cam_noisy_params['SonyA7S2_lowISO'] = {
        'Kmin':-0.2734, 'Kmax':0.64185, 'lam':-0.05, 'q':1/(2**14), 'wp':16383,
        'sigTLk':0.75004, 'sigTLb':0.88237,   'sigTLsig':0.02526,
        'sigRk':0.73954,  'sigRb':-0.32404,   'sigRsig':0.03596,
    }
    cam_noisy_params['SonyA7S2_highISO'] = {
        'Kmin':0.41878, 'Kmax':1.1234, 'lam':-0.05, 'q':1/(2**14), 'wp':16383,
        'sigTLk':0.55284, 'sigTLb':0.12758,   'sigTLsig':0.00733,
        'sigRk':0.50505,  'sigRb':-1.39476,   'sigRsig':0.02262,
    }
    cam_noisy_params['gain_0_5'] = {
    'Kmin':0.10548, 'Kmax':0.15938, 'lam':-0.100, 'q':0.000244, 'wp':1023,
    'sigRk':0.51813,  'sigRb':0.01496,  'sigRsig':0.00203,
    'sigGsk':4.35421, 'sigGsb':0.18369, 'sigGssig':0.00075,
    'sigReadk':4.38362, 'sigReadb':0.15859, 'sigReadsig':0.00053}
    cam_noisy_params['gain_5_10'] = {
        'Kmin':0.15938, 'Kmax':0.23973, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.20312,  'sigRb':0.12991,  'sigRsig':0.00223,
        'sigGsk':1.57369, 'sigGsb':0.62684, 'sigGssig':0.00062,
        'sigReadk':1.62711, 'sigReadb':0.59792, 'sigReadsig':0.00046}
    cam_noisy_params['gain_10_15'] = {
        'Kmin':0.23973, 'Kmax':0.38181, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.01324,  'sigRb':0.08439,  'sigRsig':0.00227,
        'sigGsk':-1.65368, 'sigGsb':1.40053, 'sigGssig':0.00077,
        'sigReadk':-1.68672, 'sigReadb':1.39234, 'sigReadsig':0.00052}
    cam_noisy_params['gain_15_20'] = {
        'Kmin':0.38181, 'Kmax':0.54644, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.24556,  'sigRb':-0.01442,  'sigRsig':0.00231,
        'sigGsk':1.17159, 'sigGsb':0.32180, 'sigGssig':0.00082,
        'sigReadk':1.17438, 'sigReadb':0.29993, 'sigReadsig':0.00065}
    cam_noisy_params['gain_20_25'] = {
        'Kmin':0.54644, 'Kmax':0.82669, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.08820,  'sigRb':0.16795,  'sigRsig':0.00210,
        'sigGsk':0.34172, 'sigGsb':0.77528, 'sigGssig':0.00073,
        'sigReadk':0.35262, 'sigReadb':0.74897, 'sigReadsig':0.00064}
    cam_noisy_params['gain_25_30'] = {
        'Kmin':0.82669, 'Kmax':1.25000, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.01197,  'sigRb':0.10494,  'sigRsig':0.00188,
        'sigGsk':1.05672, 'sigGsb':0.18419, 'sigGssig':0.00080,
        'sigReadk':1.06995, 'sigReadb':0.15596, 'sigReadsig':0.00074}
    cam_noisy_params['gain_30_35'] = {
        'Kmin':1.25000, 'Kmax':1.90947, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.05625,  'sigRb':0.01967,  'sigRsig':0.00227,
        'sigGsk':0.80467, 'sigGsb':0.49926, 'sigGssig':0.00100,
        'sigReadk':0.80700, 'sigReadb':0.48466, 'sigReadsig':0.00098}
    cam_noisy_params['gain_35_40'] = {
        'Kmin':1.90947, 'Kmax':2.89427, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.01247,  'sigRb':0.10326,  'sigRsig':0.00261,
        'sigGsk':0.88096, 'sigGsb':0.35359, 'sigGssig':0.00140,
        'sigReadk':0.88397, 'sigReadb':0.33768, 'sigReadsig':0.00139}
    cam_noisy_params['gain_40_45'] = {
        'Kmin':2.89427, 'Kmax':4.38321, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.04290,  'sigRb':0.01520,  'sigRsig':0.00279,
        'sigGsk':0.86907, 'sigGsb':0.38800, 'sigGssig':0.00198,
        'sigReadk':0.87109, 'sigReadb':0.37497, 'sigReadsig':0.00199}
    cam_noisy_params['gain_45_50'] = {
        'Kmin':4.38321, 'Kmax':6.47191, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.04440,  'sigRb':0.00863,  'sigRsig':0.00460,
        'sigGsk':0.91257, 'sigGsb':0.19733, 'sigGssig':0.00287,
        'sigReadk':0.91435, 'sigReadb':0.18534, 'sigReadsig':0.00288}
    cam_noisy_params['gain_50_55'] = {
        'Kmin':6.47191, 'Kmax':9.63317, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.04802,  'sigRb':-0.01485,  'sigRsig':0.00677,
        'sigGsk':0.98525, 'sigGsb':-0.27306, 'sigGssig':0.00431,
        'sigReadk':0.98560, 'sigReadb':-0.27579, 'sigReadsig':0.00429}
    cam_noisy_params['gain_55_60'] = {
        'Kmin':9.63317, 'Kmax':14.41878, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.03001,  'sigRb':0.15864,  'sigRsig':0.00845,
        'sigGsk':0.62366, 'sigGsb':3.21019, 'sigGssig':0.00612,
        'sigReadk':0.62352, 'sigReadb':3.21224, 'sigReadsig':0.00613}
    cam_noisy_params['gain_60_65'] = {
        'Kmin':14.41878, 'Kmax':21.66092, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.00001,  'sigRb':0.59124,  'sigRsig':0.00987,
        'sigGsk':0.00320, 'sigGsb':12.15648, 'sigGssig':0.00674,
        'sigReadk':0.00320, 'sigReadb':12.15643, 'sigReadsig':0.00678}
    cam_noisy_params['gain_65_70'] = {
        'Kmin':21.66092, 'Kmax':32.99669, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.00011,  'sigRb':0.59395,  'sigRsig':0.01111,
        'sigGsk':0.00183, 'sigGsb':12.18610, 'sigGssig':0.00670,
        'sigReadk':0.00183, 'sigReadb':12.18614, 'sigReadsig':0.00671}
    cam_noisy_params['gain_70_75'] = {
        'Kmin':32.99669, 'Kmax':49.78000, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.00004,  'sigRb':0.58899,  'sigRsig':0.01086,
        'sigGsk':0.00144, 'sigGsb':12.19891, 'sigGssig':0.00679,
        'sigReadk':0.00144, 'sigReadb':12.19891, 'sigReadsig':0.00682}
    cam_noisy_params['gain_75_80'] = {
        'Kmin':49.78000, 'Kmax':75.18213, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.00017,  'sigRb':0.58238,  'sigRsig':0.01138,
        'sigGsk':0.00099, 'sigGsb':12.22138, 'sigGssig':0.00671,
        'sigReadk':0.00100, 'sigReadb':12.22091, 'sigReadsig':0.00675}
    cam_noisy_params['gain_80_85'] = {
        'Kmin':75.18213, 'Kmax':113.62996, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.00003,  'sigRb':0.59748,  'sigRsig':0.01141,
        'sigGsk':0.00067, 'sigGsb':12.24564, 'sigGssig':0.00705,
        'sigReadk':0.00067, 'sigReadb':12.24583, 'sigReadsig':0.00707}
    cam_noisy_params['gain_85_90'] = {
        'Kmin':113.62996, 'Kmax':171.82296, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.00005,  'sigRb':0.59969,  'sigRsig':0.00973,
        'sigGsk':0.00044, 'sigGsb':12.27204, 'sigGssig':0.00654,
        'sigReadk':0.00044, 'sigReadb':12.27226, 'sigReadsig':0.00655}
    cam_noisy_params['gain_90_95'] = {
        'Kmin':171.82296, 'Kmax':259.90232, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':-0.00001,  'sigRb':0.59285,  'sigRsig':0.01031,
        'sigGsk':0.00030, 'sigGsb':12.29651, 'sigGssig':0.00586,
        'sigReadk':0.00030, 'sigReadb':12.29646, 'sigReadsig':0.00589}
    cam_noisy_params['gain_95_100'] = {
        'Kmin':259.90232, 'Kmax':397.76528, 'lam':-0.100, 'q':0.000244, 'wp':1023,
        'sigRk':0.00001,  'sigRb':0.58674,  'sigRsig':0.01101,
        'sigGsk':0.00020, 'sigGsb':12.32137, 'sigGssig':0.00601,
        'sigReadk':0.00020, 'sigReadb':12.32108, 'sigReadsig':0.00603}

    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        print(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use Huawei's parameters to test.''')
        return cam_noisy_params['Huawei']

# 新噪声参数采样，用于VST变换
def sample_VST(gain=50, camera_type='gain'):
    # 获取已经测算好的相机噪声参数
    for i in range(21):
        bound = i*5
        if bound>gain:
            camera_type += f'_{bound}_{bound+5}'
            break
    params = get_camera_noisy_params(camera_type=camera_type)
    wp = params['wp']
    # 根据表格得到的噪声参数
    K = ISO2K(gain2ISO(gain))
    sigR = params['sigRk']*K + params['sigRb']
    sigGs = params['sigGsk']*K + params['sigGsb']
    sigRead = params['sigReadk']*K + params['sigReadb']
    
    return {'K':K, 'sigRead':sigRead, 'sigR':sigR, 'sigGs':sigGs, 'wp':wp}

def read_raw(file, norm=True):
    raw = np.fromfile(file, dtype=np.uint16).reshape(-1, 1440, 2560) >> 2
    if norm:
        raw = raw.astype(np.float32) / 1023
    return raw

def norm(img):
    img = (img - img.min()) / (img.max()-img.min())
    return img

def Expand(img):
    return np.stack([img]*3, axis=-1)

def visualize(flow, name='flow', show=False):
    flow = flow.astype(np.float32)
    h, w, c = flow.shape
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,2] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    if show:
        cv2.imshow(name, bgr)
    print(np.max(mag))
    return bgr

def ImgShow(img, name='img', wait=True):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def PyrShow(imgs, name='img', wait=True):
    for i in range(len(imgs)):
        cv2.imshow('{}_{}'.format(name, i), imgs[i])
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def gauss_down(img, scale=2):
    blur = cv2.GaussianBlur(img, (5,5), -1)
    down = box_down(blur, scale=scale)
    return down

def box_down(img, scale=2):
    return block_reduce(img, block_size=(scale,scale), func=np.mean)

def PyrBuild(img, depth=4):
    GauPyr = [None]*depth
    LapPyr = [None]*(depth-1)
    GauPyr[0] = img
    for i in range(depth-1):
        GauPyr[i+1] = pyrDown(GauPyr[i])
        LapPyr[i] = GauPyr[i] - pyrUp(GauPyr[i+1])
    return GauPyr, LapPyr

def PyrRecovery(LapPyr, base_img):
    depth = len(LapPyr)
    RecPyr = [None]*(depth+1)
    RecPyr[depth] = base_img
    for i in range(depth):
        RecPyr[depth-i-1] = cv2.pyrUp(RecPyr[depth-i]) + LapPyr[depth-i-1]
    return RecPyr

def pyrDown(img):
    return cv2.pyrDown(img)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return img

def pyrUp(img):
    return cv2.pyrUp(img)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return img