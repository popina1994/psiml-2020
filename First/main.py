import numpy as np
from PIL import Image
if __name__ == "__main__":
    image_path = input()
    image_file = Image.open(image_path)
    image = np.array(image_file)
    pixel = image[0, 0]
    print("Red: {0}".format(pixel[0]))
    print("Green: {0}".format(pixel[1]))
    print("Blue: {0}".format(pixel[2]))