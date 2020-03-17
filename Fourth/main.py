import os
import numpy as np
import sys
from typing import List
from PIL import Image
import pathlib
import time

NUM_COLORS = 3
TEST_PATH = str(pathlib.Path(__file__).parent.absolute()) + "\\"
#TEST_NUMBERS = range(0,8) + range(10,16)
TEST_NUMBERS = [0]
X_ID = 0
Y_ID = 1
HEIGHT_ID = 2
WEIGHT_ID = 3


def get_image(path: str):
    file = Image.open(path)
    image = np.array(file)
    return image


def get_sum_rows(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_width sums in the row-th row and starting from the col-th column
    map_sum_compressed_row = np.zeros((map_height, map_width - patch_width + 1), dtype=int)
    for row in range(map_height):
        for col in range(0, patch_width):
            pixel = map_image[row][col]
            for color in range(NUM_COLORS):
                map_sum_compressed_row[row][0] += int(pixel[color])

    for row in range(map_height):
        for col in range(patch_width, map_width):
            pixel_right = map_image[row][col]
            pixel_left = map_image[row][col - patch_width]
            map_sum_compressed_row[row][col - patch_width + 1] = map_sum_compressed_row[row][col - patch_width]
            for color in range(NUM_COLORS):
                map_sum_compressed_row[row][col - patch_width + 1] += int(pixel_right[color]) - int(pixel_left[color])

    return map_sum_compressed_row


def get_sum_cols(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_height sums in the col-th columns starting from the row-th row
    map_sum_compressed_col = np.zeros((map_height - patch_height + 1, map_width), dtype=int)
    for col in range(map_width):
        for row in range(patch_height):
            pixel = map_image[row][col]
            for color in range(NUM_COLORS):
                map_sum_compressed_col[0][col] += int(pixel[color])

    for col in range(map_width):
        for row in range(patch_height, map_height):
            pixel_up = map_image[row - patch_height][col]
            pixel_down = map_image[row][col]
            map_sum_compressed_col[row - patch_height + 1][col] = map_sum_compressed_col[row - patch_height][col]
            for color in range(NUM_COLORS):
                 map_sum_compressed_col[row - patch_height + 1][col] += int(pixel_down[color]) - int(pixel_up[color])

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
            '''
            if (row == 78) and (col == 174):
                print("Assumed", assumed)
            if (row == 225) and (col == 420):
                print("Assumed2", assumed)
            if assumed != map_compressed[row][col]:
                print("Problem complete sum badly computed:{}{}".format(row, col))
            '''
    return map_compressed


def get_patch_fun(patch_image):
    patch_height, patch_width, _ = patch_image.shape
    patch_fun = 0
    for row in range(patch_height):
        for col in range(patch_width):
            pixel = patch_image[row][col]
            for color in range(NUM_COLORS):
                patch_fun += pixel[color]
    return patch_fun


def convert_image_to_sum(map_image, patch_height, patch_width):
    map_height, map_width, _ = map_image.shape
    # Each (row, col) element contains patch_width sums in the row-th row and starting from the col-th column
    map_compressed = map_compress_image(map_image, patch_height, patch_width)
    return map_compressed


def find_similar(patch_image, map_compressed):
    patch_height, patch_width, _ = patch_image.shape
    min_diff = sys.maxsize
    left_x = -1
    left_y = -1

    patch_fun = get_patch_fun(patch_image)
    #print("Patch_FUN",patch_fun)
    mc_height, mc_width = map_compressed.shape
    # Create list of so
    for row in range(0, mc_height, 10):
        for col in range(0, mc_width, 10):
            if abs(patch_fun - map_compressed[row][col]) < min_diff:
                min_diff = abs(patch_fun - map_compressed[row][col])
                left_x = col
                left_y = row


    '''
    for row_start in range(0, map_height - patch_height):
        for col_start in range(0, map_width - patch_width):
            cur_diff = 0
            for row_patch, row_map in enumerate(range(row_start, row_start + patch_height)):
                for col_patch, col_map in enumerate(range(col_start, col_start + patch_width)):
                    map_pixel = map_image[row_map, col_map]
                    patch_pixel = patch_image[row_patch, col_patch]
                    for color_id in range(0, NUM_COLORS):
                        cur_diff += abs(int(map_pixel[color_id]) - int(patch_pixel[color_id]))
                if row_map == 0:
                    print(row_map)

            if (cur_diff < max_diff):
                max_diff = cur_diff
                left_x = col_map
                left_y = row_map
    '''

    return left_x, left_y


def run_test(map_path: str, patches_path_a: List[str]):
    patch_image_a = [np.array([1])] * patches_path_a.__len__()
    map_image = get_image(map_path)

    for patch_idx, patch_path in enumerate(patches_path_a):
        patch_image_a[patch_idx] = get_image(patch_path)

    map_height,map_width, _ = map_image.shape
    patch_height, patch_width, _ = patch_image_a[0].shape
    map_compressed = convert_image_to_sum(map_image, patch_height, patch_width)
    for patch_image in patch_image_a:
        x, y = find_similar(patch_image, map_compressed)
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


def submision_test():
    map_path = input()
    num_patches = int(input())
    boring = input()
    patch_path_a = [""] * num_patches
    for patch_idx in range(num_patches):
        patch_path_a[patch_idx] = input()
    run_test(map_path, patch_path_a)

if __name__ == "__main__":

    #run_tests()
    submision_test()


