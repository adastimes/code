from PIL import Image
import numpy as np
import os

import ctypes
import numpy
import glob

# find the shared library, the path depends on the platform and Python version
libfile = '/mnt/nvme/adas/code/ctypes/build/lib.linux-x86_64-3.8/yuv2rgb.cpython-38-x86_64-linux-gnu.so'

# 1. open the shared library
yuvlib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function mysum
yuvlib.yuv2rgb.argtypes = [ctypes.c_int, ctypes.c_int,
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32),
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32),
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32),
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32),
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32),
                           numpy.ctypeslib.ndpointer(dtype=numpy.int32)]


def progress(percent=0, width=30):
    """
    Print a progress bar. We use this around the place to see where we are.
    :param percent: percent done
    :param width: width of the progress bar
    """
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)


def list_files(directory, text_filter):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            if filename.find('.bin') > 0:  # get only the png files
                if filename.find(text_filter) > 0:
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)  # Add it to the list.
    return file_paths


def y_to_img(file_in, file_out):
    """
    read luma file and save it as png
    :param file_in: the input file location
    :param file_out: the output file location
    """
    f = open(file_in, "rb")
    data = list(f.read())
    f.close()
    data_int = np.array(data, dtype='uint8')
    data_int_reshape = data_int.reshape((480, 640))
    data_int_reshape = np.transpose(data_int_reshape)
    data_int_reshape = np.flip(data_int_reshape, 1)

    img = Image.fromarray(data_int_reshape)
    img.save(file_out)
    # img.show()


def yuv_to_rgb(file_in_y, file_in_u, file_in_v, file_out):
    """
    Read the y,u,v files from kotlin and make an rgb png file to be used to to some annotation.
    :param file_in_y: y file path
    :param file_in_u: u file path
    :param file_in_v: v file path
    :param file_out: output png file
    """
    w = 480
    h = 640

    f = open(file_in_u, "rb")
    u = list(f.read())
    f.close()

    f = open(file_in_v, "rb")
    v = list(f.read())
    f.close()

    f = open(file_in_y, "rb")
    y = list(f.read())
    f.close()

    u = np.array(u, dtype=np.int32)
    v = np.array(v, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    index = np.arange(0, np.int(w * h / 2), 2, dtype=np.int)
    v = v[index]
    u = u[index]

    r = np.ndarray(w * h, dtype=np.int32)
    g = np.ndarray(w * h, dtype=np.int32)
    b = np.ndarray(w * h, dtype=np.int32)

    yuvlib.yuv2rgb(w, h, y, u, v, r, g, b)

    r = np.flip(np.transpose(r.reshape((w, h))), 1)
    g = np.flip(np.transpose(g.reshape((w, h))), 1)
    b = np.flip(np.transpose(b.reshape((w, h))), 1)

    rgb = np.zeros(shape=(h, w, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    img = Image.fromarray(np.uint8(rgb.clip(0, 255)))
    # img.show()
    img.save(file_out)


def yuv_to_rgb_python(file_in_y, file_in_u, file_in_v, file_out):
    """
    Read the y,u,v files from kotlin and make an rgb png file to be used to to some annotation.
    :param file_in_y: y file path
    :param file_in_u: u file path
    :param file_in_v: v file path
    :param file_out: output png file
    """
    # |---- Pixel stride ----|                    Row ends here --> |
    # | Pixel 1 | Other Data | Pixel 2 | Other Data | ... | Pixel N |
    w = 480
    h = 640

    w_2 = 240
    h_2 = 320

    f = open(file_in_u, "rb")
    data = list(f.read())
    f.close()

    data2 = []
    for i in range(len(data)):
        if i % 2 == 0:
            data2.append(data[i])

    data = data2

    data_int = np.array(data, dtype='uint8')
    data_int_reshape = data_int.reshape((w_2, h_2))
    data_int_reshape = np.transpose(data_int_reshape)
    u = np.flip(data_int_reshape, 1).repeat(2, axis=0).repeat(2, axis=1)

    f = open(file_in_v, "rb")
    data = list(f.read())
    f.close()
    data2 = []
    for i in range(len(data)):
        if i % 2 == 0:
            data2.append(data[i])

    data = data2
    data_int = np.array(data, dtype='uint8')
    data_int_reshape = data_int.reshape((w_2, h_2))
    data_int_reshape = np.transpose(data_int_reshape)
    v = np.flip(data_int_reshape, 1).repeat(2, axis=0).repeat(2, axis=1)

    f = open(file_in_y, "rb")
    data = list(f.read())
    f.close()
    data_int = np.array(data, dtype='uint8')
    data_int_reshape = data_int.reshape((w, h))
    data_int_reshape = np.transpose(data_int_reshape)
    y = np.flip(data_int_reshape, 1)

    array_sum = yuvlib.yuv2rgb(h, w, y, u, v, r, g, b)

    rgb = np.zeros(shape=(h, w, 3))

    for i in range(h):
        for j in range(w):
            rgb[i, j, 0] = y[i, j] + (1.370705 * (v[i, j] - 128))
            rgb[i, j, 1] = y[i, j] + (0.698001 * (v[i, j] - 128)) - (0.337633 * (u[i, j] - 128))
            rgb[i, j, 2] = y[i, j] + (1.732446 * (u[i, j] - 128))

    img = Image.fromarray(np.uint8(rgb.clip(0, 255)))
    img.save(file_out)


# get all files in the directory that have _y
# Note that files are saved in the following format:
# <burst_number>_<frame_number>_<time_between_frames>_<y,u or v>.bin
L = list_files('/home/robert/AndroidStudioProjects/files', '_y')

total_files = len(L)
for files_processed in range(total_files):
    file_bin_y = L[files_processed]
    file_bin_u = file_bin_y.replace('_y', '_u')  # the u file name
    file_bin_v = file_bin_y.replace('_y', '_v')  # the v file name

    progress(percent=round(100 * float(files_processed+1) / float(total_files)))
    file_png = file_bin_y.replace('.bin', '.png') # the output file name
    yuv_to_rgb(file_bin_y, file_bin_u, file_bin_v, file_png) #transform all
