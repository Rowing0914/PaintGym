# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# num_samples = 128
#
# if not os.path.exists("random"):
#     os.makedirs("random")
#
# img = np.random.randint(low=0, high=256, size=(num_samples, 128, 128))
#
# for img_id in range(num_samples):
#     print(img_id)
#     plt.imshow(img[img_id, :, :])
#     plt.savefig(f"./data/random/{img_id}.jpg")
#     plt.clf()

# import torchvision
#
# torchvision.datasets.MNIST(root="./images", download=True)

import os
import struct

from array import array
from os import path

import png


# source: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
def read(dataset="training", path="."):
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir):
    # create output directories
    output_dirs = [path.join(output_dir, str(i)) for i in range(10)]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".png")
        print("writing " + output_filename)
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [data[(i * rows * cols + j * cols): (i * rows * cols + (j + 1) * cols)] for j in range(rows)]
            w.write(h, data_i)


if __name__ == "__main__":
    # input_path, output_path = sys.argv[1], sys.argv[2]
    input_path, output_path = "./images/MNIST/raw", "./images/MNIST/new"

    for dataset in ["training", "testing"]:
        labels, data, size, rows, cols = read(dataset, input_path)
        write_dataset(labels, data, size, rows, cols,
                      path.join(output_path, dataset))
