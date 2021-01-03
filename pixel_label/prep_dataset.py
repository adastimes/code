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
import csv
from common import *
from PIL import Image


class ImageResizer:

    def __init__(self, directory, text_filter='small_', resize_factor=2, delete_files=True):
        """
        Initialize the parameters of the class and resize the dataset. Only the files that do not contain
        a filter are resized. The resized files will have the filter string added and will be save in the same
        directory.

        :param directory: This is the directory that is parsed where images are searched
        :param text_filter: this is a text filter that is used to discriminate the files. If file contains
        the string it is omitted.The same filter is added to the new files produced, at the beginning.Default
        is "small_".
        :param resize_factor: The reiszing factor is for both x and y direction. Default is 2.
        :param delete_files: if true it will delete firs the files that contain the text_filter.Default is true.
        """
        self.file_paths = []
        self.new_file_paths = []  # after resizing we will put the new pats and names in this list
        self.directory = directory
        self.text_filter = text_filter
        self.resize_factor = resize_factor
        self.delete_files = delete_files

    def get_images_from_dir(self):
        """
        This function will generate the file names in a directory
        tree by walking the tree either top-down or bottom-up. For each
        directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        self.file_paths = []  # List which will store all of the full filepaths.

        # Walk the tree.
        for root, directories, files in os.walk(self.directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                if filename.find('.png') > 0:  # get only the png files
                    if filename.find(self.text_filter) == -1:  # do not get the files that contain the filter
                        filepath = os.path.join(root, filename)
                        self.file_paths.append(filepath)  # Add it to the list.

    def im_resize(self, input_file_name, output_file_name):
        """
        This function reads an image , downsamples the content with nearest with a fator specified by
         self.resize_factor and then writes it out in another file.
        :param input_file_name: input file path
        :param output_file_name: output file path
        """
        try:
            im = Image.open(input_file_name)
            width = int(im.size[0] / self.resize_factor)
            height = int(im.size[1] / self.resize_factor)
            tmp = im.resize((width, height), Image.NEAREST)

        except:
            print("Failed:", input_file_name)

        tmp.save(output_file_name)

    def delete_files_with_filter(self):
        """
            This function should delete all .png files that begin with self.text_filter in a directory structure.
        """
        # Walk the tree.
        nr_files_deleted = 0
        for root, directories, files in os.walk(self.directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                if filename.find(self.text_filter) == 0:  # get only files that begin with text filter
                    if filename.find('.png') > 0:  # get only the png files
                        os.remove(filepath)
                        nr_files_deleted = nr_files_deleted + 1

        print(" Nr. deleted files:", nr_files_deleted)

    def resize_all_files(self):
        """
        This function will resize all the images in the self.file_paths list.
        """
        current_file = 0
        for element in self.file_paths:
            current_file += 1
            # rename the file
            element_parts = element.split('/')  # split into string based on separator /
            element_parts[element_parts.__len__() - 1] = self.text_filter + element_parts[
                element_parts.__len__() - 1]  # last should be the file name
            # form again the full path
            new_file_location = ''
            for i in element_parts:
                new_file_location = new_file_location + '/'
                new_file_location = new_file_location + i

            self.new_file_paths.append(new_file_location)
            self.im_resize(element, new_file_location)
            progress(int(100 * current_file / len(self.file_paths)))

    def write_csv(self, file_name, file_filter):
        """
        write a csv file with all the images in the new list (the one containing the resized images).
        Only file names that contain som filter text are added.
        :param file_name: the location of the csv file
        :param file_filter: the text that the file path contains
        """
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["id", "FileName"])
        file_id = 0
        for files in self.new_file_paths:
            if files.find(file_filter) >= 0:
                if os.path.isfile(files):
                    writer.writerow([file_id, files])
                    file_id += 1
        csv_file.close()

    def run(self):
        """
            Run everything. Find .png files, resize them and write them back.
        """
        print("\nProcessing the directory: ", self.directory)
        if self.delete_files:  # should we delete the things ?
            self.delete_files_with_filter()
        self.get_images_from_dir()
        self.resize_all_files()


ref = ImageResizer(reference_dir, resize_factor=4, delete_files=True)
data = ImageResizer(dataset_dir, resize_factor=4, delete_files=True)
ref.run()
data.run()

data.write_csv(dataset_dir + '/train.csv', '/train')
data.write_csv(dataset_dir + '/val.csv', '/val')
data.write_csv(dataset_dir + '/test.csv', '/test')

print("\nDone preparing the dataset!")
