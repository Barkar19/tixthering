#!/usr/bin/python3

import cv2
import sys
import numpy as np
# from skimage.util.shape import view_as_windows
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import numbers
from numpy.lib.stride_tricks import as_strided
import string


def view_as_windows(arr_in, window_shape, step=1):
    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    if not arr_in.flags.contiguous:
        warn(RuntimeWarning("Cannot provide views on a non-contiguous input "
                            "array without copying."))

    arr_in = np.ascontiguousarray(arr_in)

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

def text_phantom(text, size):
    # Availability is platform dependent

    # Create font
    pil_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size=size // len(text),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width) // 2,
              (size - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    arr = (255 - np.asarray(canvas)) / 255.0
    arr[arr < 0.2] = 0
    return bbox2(arr)[:,:,1] * 255


max_width = -1
max_height = -1
digits_patches = []
for n in "0123456789":
    arr = text_phantom(str(n), int(sys.argv[1]))
    digits_patches.append(arr)
    # print(arr)
    # print(arr.shape)
    max_width = max(max_width, arr.shape[0])
    max_height = max(max_height, arr.shape[1])
    # print(max_width)
    # print(max_height)
    # plt.imshow(arr)
    # plt.waitforbuttonpress()
result = []
for n in digits_patches:
    out = None
    if n.shape[0] < max_width:
        tmp = np.zeros((max_width-n.shape[0],n.shape[1]))
        # tmp = np.expand_dims(tmp, axis=0)
        # print(tmp)
        # print(n)
        n = np.concatenate((n,tmp), axis=0)
    if n.shape[1] < max_height:
        # print(max_height-n.shape[1])
        tmp = np.zeros((n.shape[0],max_height-n.shape[1]))
        # tmp = np.expand_dims(tmp, axis=0)
        # print(n.shape[1], max_height)
        # print(tmp)
        n = np.concatenate((n,tmp), axis=1)
    result.append(n)


# N = int(sys.argv[2])
#
# r = 256 / (N-1)
# ranges = dict()
# ranges[0] = (0,r/2)
# value = ranges[0][1]
# for n in range(1,N):
#     # last = ranges[-1]
#     # value = value - 1
#     if n == N-1:
#         key = int(255)
#     else:
#         key = int(round(value + r / 2))
#     new = (value, value + r)
#     ranges[key] = new
#     value = value + r
# ranges[255] = (256 - r//2, 256)
# ranges[0] = (0,r//2)

# print(result)
# exit(0)
# available_patches = result
# available_patches = [ 255 - x for x in result]
available_patches = result + [ 255 - x for x in result]
# [
#     [[255, 128, 255],
#      [128, 255, 128],
#      [255, 128, 255]],
#
#     [[64, 32, 0],
#      [128, 64, 32],
#      [255, 128, 64]],
#
#     [[0, 0, 32],
#      [0, 32, 64],
#      [32, 64, 32]],
#
#     [[32, 80, 40],
#      [160, 32, 80],
#      [255, 160, 32]],
#
#     [[40, 128, 255],
#      [40, 128, 255],
#      [40, 128, 255]],
# ]
def find_closest_palette_color( value ):
    if value <= 0:
        return 0
    elif value > 255.0:
        return 255.0
    else:
        for key,rng in ranges.items():
            # print(round(value))
            if rng[0] < value and value <= rng[1]:
                # print(key, value)
                return key
    print("Error!", value)
    exit(1)
    return 0

def find_closest_patch(value):
    min = (available_patches[0], sys.float_info.max)
    # print("new")
    for p in available_patches:
        # print("P:", p)
        # print("V", value)
        mse = ((p - value)**2).mean()
        # print(mse)
        # print(min[0])
        if( min[1] > mse ):
            min = (p, mse)
    return np.array(min[0])

def set_output(output, idx, window, value):
    for n in np.ndindex(window):
        # print(value)
        coords = (idx[0]*window[0] + n[0], idx[1]*window[1] + n[1])
        # print("Coords", coords)
        # print('Val', value[n])

        output[coords] = value[n]

def apply_dithering( image, pattern ):
    output = np.zeros_like(image)
    window_shape = available_patches[0].shape
    # print(isinstance(window_shape, numbers.Number))
    # window_shape = (window_shape[1],window_shape[0])
    patches = view_as_windows( image, window_shape, step = window_shape )
    # output = output.reshape((output.shape[0] // window_shape[0],
    #                 output.shape[1] // window_shape[1],
    #                 window_shape[0],
    #                 window_shape[1]))
    # print(image.shape)
    # print(patches.shape)

    patch_width = patches.shape[0]
    patch_height = patches.shape[1]

    coords = np.where(pattern == -1)
    coords = (-coords[0][0], -coords[1][0])
    all_patches = patches.shape[0] * patches.shape[1]
    for enum, current_idx in enumerate(np.ndindex(patches.shape[0:2])):
        sys.stdout.write("\r")
        sys.stdout.write("{:02.2f} % ".format(100*enum / all_patches))
        old_value = patches[current_idx]
        new_value = find_closest_patch(old_value)
        # print( "SHL", output.shape )
        error = old_value - new_value
        # print(output[current_idx])
        set_output(output,current_idx,window_shape,new_value)
        # print("For current")
        # print( current_idx )
        # print( patches[current_idx] )
        # print("Err")
        # print(error)
        # p = np.copy(patches)
        # print("CurrentBef")
        # print(p.transpose(0,1,3,2).reshape(image.shape))
        for next_idx in np.ndindex(pattern.shape[0:2]):
            # print("next", next_idx, pattern[next_idx])
            if pattern[next_idx] > 0:
                # next_idx = (next_idx[1], next_idx[0])
                new_idx = tuple([sum(x) for x in zip(current_idx, next_idx, coords)])
                # print("New: ", new_idx)
                if 0 <= new_idx[0] < patch_width and \
                   0 <= new_idx[1] < patch_height:
                    # pass
                    # print(new_idx)
                    # print("Before")
                    # print(patches[new_idx])
                    patches[new_idx] = patches[new_idx] + error * pattern[next_idx]
                    # print("After")
                    # print(patches[new_idx])
        # p = np.copy(patches)
        # print("Current")
        # print(p.reshape(image.shape))
        # o = np.copy(output)
        # print("Output")
        # print(o)


    return output.reshape(image.shape)


img = cv2.imread(sys.argv[2])

img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
touch = np.ndarray(shape=img.shape, dtype=np.uint8)
# print(img.shape)


sample = np.copy(img[0:6,0:6])

sample[0:2,0:2] = 0
sample[2:4,2:4] = 0
sample[4:6,4:6] = 0
# print(sample)
pattern = np.array([[0, -1, 7/16],
                    [3/16, 5/16, 1/16]])
pattern.transpose((1,0))
out = apply_dithering(img, pattern)
# print(out)


# # img = img / 255.0
# for (x,y), value in np.ndenumerate(img):
#   # print(x,y,value)
#   newpixel = find_closest_palette_color(value)
#   img[x,y] = newpixel
#   touch[x,y] = newpixel
#   # if newpixel != 0.0 and newpixel != 1.0:
#   # print(newpixel)
#   quant_error = value - newpixel
#
#   if(x+1 <= img.shape[0]-1 ):
#     img[x + 1][y] = img[x + 1][y] + quant_error * 7 / 16
#   if x-1 >= 0 and y+1 <= img.shape[1]-1:
#     img[x - 1][y + 1] = img[x - 1][y + 1] + quant_error * 3 / 16
#   if y + 1 <= img.shape[1] - 1:
#     img[x][y + 1] = img[x][y + 1] + quant_error * 5 / 16
#   if x+1 <= img.shape[0]-1 and y + 1 <= img.shape[1]-1:
#     img[x + 1][y + 1] = img[x + 1][y + 1] + quant_error * 1 / 16
#   # print("VAL:", img[x,y])
# # print(img)
# # img = img * 255
# img = np.uint8(img)
# # for (x,y), value in np.ndenumerate(img):
#   # if value != 0 and value != 255:
#   # print("NEW:", x,y,value)
# print(np.histogram(img,bins=4))
# print(np.unique(touch, return_counts=True))
#
#
cv2.imwrite(sys.argv[3], out)
print()