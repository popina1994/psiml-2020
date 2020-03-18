import os
import numpy as np
from typing import List
from PIL import Image
import pathlib
import time

NUM_COLORS = 3
TEST_PATH = str(pathlib.Path(__file__).parent.absolute()) + "\\"
#TEST_NUMBERS = range(6)
TEST_NUMBERS = [5]
X_ID = 0
Y_ID = 1
HEIGHT_ID = 2
WEIGHT_ID = 3
FUN_ID = 0
ROW_ID = 1
COL_ID = 2



def get_image(path: str):
    file = Image.open(path)
    image = np.array(file)
    return image

def gray_scale_pixel(pixel):
    return 0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2]


def rgb_to_hsl(pixel):
    red_f = pixel[0] / 255
    green_f = pixel[1] / 255
    blue_f = pixel[2] / 255

    max_color = max(red_f, max(green_f, blue_f))
    min_color = min(red_f, min(green_f, blue_f))

    if (red_f == green_f) and (green_f == blue_f):
        hue = 0
        saturation = 0
        lightness = red_f
    else:
        lightness = (min_color + max_color) / 2
        diff_color = max_color - min_color
        if (lightness < 0.5):
            saturation = diff_color / (max_color + min_color)
        else:
            saturation = diff_color / (2.0 - max_color - min_color)
        if red_f == max_color:
            hue = (green_f - blue_f) / (max_color - min_color)
        elif green_f == max_color:
            hue = 2.0 + (blue_f - red_f) / (max_color - min_color)
        else:
            hue = 4.0 + (red_f - green_f) / (max_color - min_color)
        hue /= 6
        if hue < 0:
            hue += 1


    hue = int(hue * 360)
    saturation = int(saturation * 255)
    lightness = int(lightness * 255)
    return hue, saturation, lightness


def saturatepicture(image):
    pass


def get_sum_rows(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_width sums in the row-th row and starting from the col-th column
    map_sum_compressed_row = np.zeros((map_height, map_width - patch_width + 1), dtype=int)
    for row in range(map_height):
        for col in range(0, patch_width):
            pixel = map_image[row][col]
            for color in range(NUM_COLORS):
                map_sum_compressed_row[row][0] += int(pixel[color])* int(pixel[color])

    for row in range(map_height):
        for col in range(patch_width, map_width):
            pixel_right = map_image[row][col]
            pixel_left = map_image[row][col - patch_width]
            map_sum_compressed_row[row][col - patch_width + 1] = map_sum_compressed_row[row][col - patch_width]
            for color in range(NUM_COLORS):
                map_sum_compressed_row[row][col - patch_width + 1] += int(pixel_right[color]) * int(pixel_right[color]) - \
                                                                      int(pixel_left[color]) * int(pixel_left[color])

    return map_sum_compressed_row


def get_sum_cols(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_height sums in the col-th columns starting from the row-th row
    map_sum_compressed_col = np.zeros((map_height - patch_height + 1, map_width), dtype=int)
    for col in range(map_width):
        for row in range(patch_height):
            pixel = map_image[row][col]
            for color in range(NUM_COLORS):
                map_sum_compressed_col[0][col] += int(pixel[color]) * int(pixel[color])

    for col in range(map_width):
        for row in range(patch_height, map_height):
            pixel_up = map_image[row - patch_height][col]
            pixel_down = map_image[row][col]
            map_sum_compressed_col[row - patch_height + 1][col] = map_sum_compressed_col[row - patch_height][col]
            for color in range(NUM_COLORS):
                 map_sum_compressed_col[row - patch_height + 1][col] += int(pixel_down[color]) * int(pixel_down[color]) - \
                                                                        int(pixel_up[color]) * int(pixel_up[color])

    return map_sum_compressed_col


def map_compress_image(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    map_sum_compressed_row = get_sum_rows(map_image, patch_height, patch_width)
    map_sum_compressed_col = get_sum_cols(map_image, patch_height, patch_width)

    mscr_height, mscr_width  = map_sum_compressed_row.shape
    mscc_height, mscc_width = map_sum_compressed_col.shape
    '''
    # Check for rows:
    for row in range(mscr_height):
        for col in range(mscr_width):
            assumed = np.sum(map_image[row, col:col+patch_width,:])
            if assumed != map_sum_compressed_row[row][col]:
                print ("Problem row sum badly computed:{}{}".format(row, col))
    # Check for cols:
    for row in range(mscc_height):
        for col in range(mscc_width):
            assumed = np.sum(map_image[row:row+patch_height, col,:])
            if assumed != map_sum_compressed_col[row][col]:
                print ("Problem col sum badly computed:{}{}".format(row, col))
'''
    map_compressed = np.zeros((map_height - patch_height + 1, map_width - patch_width + 1), dtype=int)
    for row in range(patch_height):
        map_compressed[0][0] += map_sum_compressed_row[row][0]

    for row in range(1, map_height - patch_height + 1):
        map_compressed[row][0] = map_compressed[row - 1][0] + map_sum_compressed_row[row + patch_height - 1][0] - \
                                 map_sum_compressed_row[row - 1][0]

    for row in range (0, map_height - patch_height + 1):
        for col in range(1, map_width - patch_width + 1):
            map_compressed[row][col] = map_compressed[row][col-1] + \
                                       map_sum_compressed_col[row][col + patch_width - 1] - \
                                       map_sum_compressed_col[row][col-1]

    # Check for rows:
    for row in range(map_compressed.shape[0]):
        for col in range(map_compressed.shape[1]):
            assumed = np.sum(map_image[row:row + patch_height, col:col + patch_width, :])
    return map_compressed


def get_patch_fun(patch_image):
    patch_height, patch_width, _ = patch_image.shape
    patch_fun = 0
    for row in range(patch_height):
        for col in range(patch_width):
            pixel = patch_image[row][col]
            for color in range(NUM_COLORS):
                patch_fun += int(pixel[color]) * int(pixel[color])
    return patch_fun


def convert_image_to_sum(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_width sums in the row-th row and starting from the col-th column
    map_compressed = map_compress_image(map_image, patch_height, patch_width)
    return map_compressed


def sort_image_compressed(map_compressed):
    mc_height, mc_width = map_compressed.shape
    map_fun_coordinate = []
    for row in range(mc_height):
        for col in range(mc_width):
            map_fun_coordinate.append((map_compressed[row][col], row, col))

    map_fun_coordinate = sorted(map_fun_coordinate, key=lambda tup: tup[0])
    return map_fun_coordinate



def find_closest_patch(map_fun_coordinate, patch_fun):
    start_idx = 0
    end_idx = map_fun_coordinate.__len__()
    mid_tuple = (0, -1, -1)
    while (end_idx - start_idx) > 1:
        mid_idx = (start_idx + end_idx) // 2
        mid_tuple = map_fun_coordinate[mid_idx]
        if (mid_tuple[FUN_ID] == patch_fun):
            return mid_tuple
        if (mid_tuple[FUN_ID] < patch_fun):
            start_idx = mid_idx
        else:
            end_idx = mid_idx

    if map_fun_coordinate[start_idx][FUN_ID] < map_fun_coordinate[end_idx][FUN_ID]:
        return map_fun_coordinate[start_idx]
    else:
        return map_fun_coordinate[end_idx]


def find_similar(patch_image, map_fun_coordinate):
    patch_height, patch_width, _ = patch_image.shape

    patch_fun = get_patch_fun(patch_image)
    closest_value, row, col = find_closest_patch(map_fun_coordinate, patch_fun)
    #print("Values: ", closest_value, " ",  patch_fun)
    left_x = col
    left_y = row
    '''
    for row in range(mc_height):
        for col in range(mc_width):
            if abs(patch_fun - map_compressed[row][col]) < min_diff:
                min_diff = abs(patch_fun - map_compressed[row][col])
                left_x = col
                left_y = row
    '''
    the_same = True if closest_value == patch_fun else False
    return left_x, left_y, the_same


def blur_patch(patch_image, blurred_patch_image):

    patch_height, patch_width, _ = blurred_patch_image.shape
    window_height = 3
    window_width = 3
    edge_height = window_height // 2
    edge_width = window_width // 2
    indexes_array = [(0, edge_height, 0, patch_width),
                   (patch_height - edge_height, patch_height, 0, patch_width),
                   (0, patch_height, 0 , edge_width),
                   (0, patch_height, patch_width - edge_width, patch_width )]
    for indexes in indexes_array:
        for row in range(indexes[0], indexes[1]):
            for col in range(indexes[2], indexes[3]):
                for color in range(NUM_COLORS):
                    blurred_patch_image[row][col][color] = patch_image[row][col][color]

    median_a = np.zeros((window_height, window_width))
    for row in range(edge_height, patch_height - edge_height):
        for col in range(edge_width, patch_width - edge_width):
            for color in range(NUM_COLORS):
                sum = 0
                # TODO: Compute median filter

                for window_row in range(0, window_height):
                    for window_col in range(0, window_width):
                        median_a[window_row][window_col] = patch_image[row + window_row - edge_height][col + window_col - edge_width][color]
                #assumed_c = np.median(patch_image[row-edge_height:row+edge_height+1, col - edge_width : col + edge_width+1, color])
                #if (assumed_c != sum):
                #    print("ROW, {} COL {} color {}, assumed_c {} sum {}".format(row, col, color, assumed_c, sum))
                blurred_patch_image[row][col][color] = np.median(median_a)

    #print("EHsd")

def run_test(map_path: str, patches_path_a: List[str]):
    patch_image_a = [np.array([1])] * patches_path_a.__len__()
    map_image = get_image(map_path)

    for patch_idx, patch_path in enumerate(patches_path_a):
        patch_image_a[patch_idx] = get_image(patch_path)

    map_height,map_width, _ = map_image.shape
    patch_height, patch_width, _ = patch_image_a[0].shape

    map_compressed = convert_image_to_sum(map_image, patch_height, patch_width)
    map_fun_coordinate = sort_image_compressed(map_compressed)

    blurred_patch_image = np.zeros(patch_image_a[0].shape, dtype=np.uint8)
    blurred_map_image = np.zeros(map_image.shape, dtype=np.uint8)

    for patch_image_idx, patch_image in enumerate(patch_image_a):
        x, y, the_same = find_similar(patch_image, map_fun_coordinate)
        if not the_same:
            print("PIC:", patches_path_a[patch_image_idx])
            img1 = Image.fromarray(patch_image, 'RGB')
            img1.show()
            blur_patch(patch_image, blurred_patch_image)
            img2 = Image.fromarray(blurred_patch_image, 'RGB')
            img2.show()
            img2.save("test.png")
            x, y, the_same = find_similar(patch_image, map_fun_coordinate)
        print("{},{}".format(x, y))


def run_tests():
    map_path  = TEST_PATH + "map.png"
    input_folder_a = []
    for test_number in TEST_NUMBERS:
        input_folder_a.append(TEST_PATH + str(test_number))

    # Run test cases.
    for input_folder in input_folder_a:
        print("Test path:" + input_folder)

        start_time = time.time()
        patches_path_a = os.listdir(input_folder)
        tmp_a = [os.path.join(input_folder, patches_path) for patches_path in patches_path_a]
        run_test(map_path, tmp_a)
        print("--- %s seconds ---" % (time.time() - start_time))


def submission_test():
    map_path = input()
    num_patches = int(input())
    boring = input()
    patch_path_a = [""] * num_patches
    for patch_idx in range(num_patches):
        patch_path_a[patch_idx] = input()
    run_test(map_path, patch_path_a)


if __name__ == "__main__":

    run_tests()
    #submission_test()


