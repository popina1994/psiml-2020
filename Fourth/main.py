import json
import os
import numpy as np
import sys
from typing import List
from PIL import Image

NUM_COLORS = 3
TEST_PATH = r"C:\Users\dorvic\PycharmProjects\psiml-2019\Fourth" + "\\"
#TEST_NUMBERS = range(0,8) + range(10,16)
TEST_NUMBERS = [0]
#TEST_NUMBERS = range(16)
X_ID = 0
Y_ID = 1
HEIGHT_ID = 2
WEIGHT_ID = 3


def get_image(path: str):
    file = Image.open(path)
    image = np.array(file)
    return image

def convert_image_to_sum(map_image, patch_height, patch_width):
    map_width, map_height, _ = map_image.shape
    map_sum_compressed_row = np.zeros((map_width - patch_width, map_height - patch_height))



def find_similar(patch_image, map_image):
    map_width, map_height, _ = map_image.shape
    patch_width, patch_height, _ = patch_image.shape
    max_diff = sys.maxsize
    left_x = -1
    left_y = -1
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
    print("END")
    return left_x, left_y

def run_test(map_path: str, patches_path_a: List[str]):
    patch_image_a = [np.array([1])] * patches_path_a.__len__()
    map_image = get_image(map_path)

    for patch_idx, patch_path in enumerate(patches_path_a):
        patch_image_a[patch_idx] = get_image(patch_path)
    print("TO")
    for patch_image in patch_image_a:
        x, y = find_similar(patch_image, map_image)
        print("{},{}".format(x, y))

def run_tests():
    map_path  = TEST_PATH + "map.png"
    input_folder_a = []
    for test_number in TEST_NUMBERS:
        input_folder_a.append(TEST_PATH + str(test_number))

    # Run test cases.
    for input_folder in input_folder_a:
        print("Test path:" + input_folder)
        patches_path_a = os.listdir(input_folder)
        tmp_a = [os.path.join(input_folder, patches_path) for patches_path in patches_path_a]
        run_test(map_path, tmp_a)


def submision_test():
    bb_path = input()
    joints_path = input()
    run_test(bb_path, joints_path)

if __name__ == "__main__":

    run_tests()
    #submision_test()


