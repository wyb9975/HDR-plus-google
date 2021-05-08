from utils import *
'''
 * merge_temporal -- combines aligned tiles in the temporal dimension by
 * weighting various frames based on their L1 distance to the reference frame's
 * tile. Thresholds L1 scores so that tiles above a certain distance are completely
 * discounted, and tiles below a certain distance are assumed to be perfectly aligned.
'''
def merge_temporal(imgs, width, height, frames, alignment):
    def weight(tx, ty, n):#"merge_temporal_weights"
        # weight for each tile in temporal merge inversely proportional to reference and alternate tile L1 distance
        select(norm_dist > (max_dist - min_dist), 0., 1. / norm_dist)
        return 
    def total_weight("merge_temporal_total_weights"):
    def output("merge_temporal_output"):

    r0 = np.array([0, 16, 0, 16]) # reduction over pixels in downsampled tile
    r1 = np.array([1, frames - 1])# reduction over alternate images

    # mirror input with overlapping edges

    imgs_mirror = imgs.copy()

    # downsampled layer for computing L1 distances

    layer = box_down2(imgs_mirror, "merge_layer")

    # alignment offset, indicies and pixel value expressions used twice in different reductions

    Point offset
    al_x, al_y, ref_val, alt_val

    # expressions for summing over pixels in each tile

    offset = np.clip(np.array(alignment(tx, ty, n)), MIN_OFFSET, MAX_OFFSET)

    al_x = idx_layer(tx, r0.x) + offset.x / 2
    al_y = idx_layer(ty, r0.y) + offset.y / 2

    ref_val = layer(idx_layer(tx, r0.x), idx_layer(ty, r0.y), 0)
    alt_val = layer(al_x, al_y, n)

    # constants for determining strength and robustness of temporal merge

    factor = 8.                         # factor by which inverse function is elongated
    min_dist = 10                          # pixel L1 distance below which weight is maximal
    max_dist = 300                         # pixel L1 distance above which weight is zero

    # average L1 distance in tile and distance normalized to min and factor

    dist = np.sum(np.abs(np.int(ref_val) - np.int(alt_val))) / 256

    norm_dist = max(1, np.int(dist) / factor - min_dist / factor)

    # total weight for each tile in a temporal stack of images

    total_weight(tx, ty) = sum(weight(tx, ty, r1)) + 1.              # additional 1. accounting for reference image

    # expressions for summing over images at each pixel

    offset = np.array(alignment(tx, ty, r1))

    al_x = idx_im(tx, ix) + offset.x
    al_y = idx_im(ty, iy) + offset.y

    ref_val = imgs_mirror(idx_im(tx, ix), idx_im(ty, iy), 0)
    alt_val = imgs_mirror(al_x, al_y, r1)

    # temporal merge function using weighted pixel values

    output(ix, iy, tx, ty) = sum(weight(tx, ty, r1) * alt_val / total_weight(tx, ty)) + ref_val / total_weight(tx, ty)

    return output

# /*
#  * merge_spatial -- smoothly blends between half-overlapped tiles in the spatial
#  * domain using a raised cosine filter.
#  */
def merge_spatial(input):

    # weight("raised_cosine_weights")
    # output("merge_spatial_output")

    Var v, x, y

    # (modified) raised cosine window for determining pixel weights

    pi = 3.1415926
    weight(v) = 0.5 - 0.5 * np.cos(2 * pi * (v + 0.5) / T_SIZE)

    # tile weights based on pixel position

    weight_00 = weight(idx_0(x)) * weight(idx_0(y))
    weight_10 = weight(idx_1(x)) * weight(idx_0(y))
    weight_01 = weight(idx_0(x)) * weight(idx_1(y))
    weight_11 = weight(idx_1(x)) * weight(idx_1(y))

    # values of pixels from each overlapping tile

    val_00 = input(idx_0(x), idx_0(y), tile_0(x), tile_0(y))
    val_10 = input(idx_1(x), idx_0(y), tile_1(x), tile_0(y))
    val_01 = input(idx_0(x), idx_1(y), tile_0(x), tile_1(y))
    val_11 = input(idx_1(x), idx_1(y), tile_1(x), tile_1(y))

    # spatial merge function using weighted pixel values

    output(x, y) = np.uint16(weight_00 * val_00
                     + weight_10 * val_10
                     + weight_01 * val_01
                     + weight_11 * val_11)

    return output
