from PIL import Image, ImageDraw
import numpy as np
import os

import ctypes
import numpy
import glob
from xml.dom import minidom

width = 480
height = 640


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


def list_files(directory):
    """
    Get the XML files.
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            if filename.find('.xml') > 0:  # get only the xml files
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return file_paths


def xml2png(xml_file, png_file):
    doc = minidom.parse(xml_file)
    items = doc.getElementsByTagName('pt')

    im_arr = np.zeros(shape=(height, width), dtype=np.uint8)
    img = Image.fromarray(im_arr)
    draw = ImageDraw.Draw(img)

    points = []

    for item in items:
        points.append((int(item.childNodes[0].childNodes[0].data), int(item.childNodes[1].childNodes[0].data)))

    draw.polygon(points, fill=7, outline=7)
    img.save(png_file)


L = list_files('/mnt/nvme/work/my_classifier/road2')
total_files = len(L)
for files_processed in range(total_files):
    file_xml = L[files_processed]
    progress(percent=round(100 * float(files_processed + 1) / float(total_files)))
    file_png = file_xml.replace('.xml', '.png')
    xml2png(file_xml, file_png)
