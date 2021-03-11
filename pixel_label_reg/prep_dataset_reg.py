'''
Prepare the dataset:
    - resize images from the directory
    - write csv file (train, val, test)

If memory is low then resize images to be able to run the things.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import os
from PIL import Image
import numpy as np
import psutil


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


class PrepDataset:

    def __init__(self, directory):
        """
        Initialize the parameters of the class and resize the dataset. Only the files that do not contain
        a filter are resized. The resized files will have the filter string added and will be save in the same
        directory.

        :param directory: This is the directory the directory with the references
        :param delete_files: if true it will delete firs the files that contain the text_filter.Default is true.
        """
        self.file_paths = []
        self.new_file_paths = []  # after resizing we will put the new pats and names in this list
        self.directory = directory
        self.text_filter1 = '_gtFine_labelIds'
        self.text_filter2 = 'small_'

    def get_images_from_dir(self):
        """
        Gets the list of files with the references from the cityscape dataset.
        """
        self.file_paths = []  # List which will store all of the full filepaths.

        # Walk the tree.
        for root, directories, files in os.walk(self.directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                if filename.find('.png') and filename.find('.bin') == -1:  # get only the png files
                    if (filename.find(self.text_filter1) > 0) and (
                            filename.find(self.text_filter2) > -1) and (
                            root.find('/test/') == -1):  # do not get the files that contain the filter
                        filepath = os.path.join(root, filename)
                        self.file_paths.append(filepath)  # Add it to the list.

    def run(self):
        """
            Run everything. Find .png files, resize them and write them back.
        """
        print("\nProcessing the directory: ", self.directory)
        self.get_images_from_dir()

        file_nr = 0
        total_file_nr = len(self.file_paths)

        min_val = 255
        max_val = 0
        for files in self.file_paths:
            file_nr += 1
            #files = '/mnt/nvme/work/Cityscape/gtFine/val/frankfurt/small_frankfurt_000001_049298_gtFine_labelIds.png'
            progress(np.int(file_nr * 100 / total_file_nr))
            labels = np.array(Image.open(files))
            freespace = np.zeros(shape=labels.shape[1], dtype=np.int)
            #img_name = files.replace('/gtFine/', '/leftImg8bit/').replace('_gtFine_labelIds', '_leftImg8bit')
            #img = np.array(Image.open(img_name))
            for i in range(labels.shape[1]):
                col = labels[:, i]
                ignore = [0, 1, 2, 3, 7]

                for j in range(len(col) - 1, 0, -1):
                    if col[j] not in ignore:
                        freespace[i] = j
                        break

            files_new = files + '.bin'
            if os.path.exists(files_new):
                os.remove(files_new)

            np.savetxt(files_new, freespace, fmt='%d')

            # img_name = files.replace('/gtFine/', '/leftImg8bit/').replace('_gtFine_labelIds', '_leftImg8bit')
            # img = np.array(Image.open(img_name))
            # for j in range(len(freespace)):
            #    img[freespace[j], j, 1] = 255

            # Image.fromarray(img).show()
            # print('aaaaaa')


reference_dir = '/mnt/nvme/work/Cityscape/gtFine/'

prep = PrepDataset(directory=reference_dir)
prep.run()
print("\nDone preparing the dataset!")
