#!/usr/bin/python3

import cv2
import sys
import numpy as np
from skimage.util.shape import view_as_windows

N = int(sys.argv[2])

r = 256 / (N-1)
ranges = dict()
ranges[0] = (0,r/2)
value = ranges[0][1]
for n in range(1,N):
    # last = ranges[-1]
    # value = value - 1
    if n == N-1:
        key = int(255)
    else:
        key = int(round(value + r / 2))
    new = (value, value + r)
    ranges[key] = new
    value = value + r
# ranges[255] = (256 - r//2, 256)
# ranges[0] = (0,r//2)

# print(ranges)
# exit(0)
available_patches = [
    [[255, 255],
     [255, 255]],
    [[128, 64],
     [0, 64]],
    [[0 , 0],
     [0, 64]],
    [[255, 0],
     [0, 255]],
    [[0, 255],
     [255, 0]],

]
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
    window_shape = (2,2)
    patches = view_as_windows( image, window_shape, step = window_shape[0] )
    # output = output.reshape((output.shape[0] // window_shape[0],
    #                 output.shape[1] // window_shape[1],
    #                 window_shape[0],
    #                 window_shape[1]))
    patch_width = patches.shape[0]
    patch_height = patches.shape[1]

    coords = np.where(pattern == -1)
    coords = (-coords[0][0], -coords[1][0])
    for current_idx in np.ndindex(patches.shape[0:2]):
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
                    pass
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


img = cv2.imread(sys.argv[1])

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