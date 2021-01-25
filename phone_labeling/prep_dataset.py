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


class PrepData:

    def __init__(self):
        """
        Write the train val tes
        :param directory: This is the directory that is parsed where images are searched
        """
        self.file_train = []
        self.file_val = []
        self.file_test = []

    def get_train_val_list(self):
        """
        Get the file list.
        """
        total_files = []
        # Walk the tree.
        for root, directories, files in os.walk(reference_dir):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                if filename.find('.png') > 0:  # get only the png files
                    filepath = os.path.join(dataset_dir + '/', filename)
                    total_files.append(filepath)  # Add it to the list.

        for files in total_files:
            if np.random.rand(1) < 0.1:  # put it in validation
                self.file_val.append(files.replace('.png', '.bin'))
            else:
                self.file_train.append(files.replace('.png', '.bin'))

    def get_test_list(self):
        """
        Get the file list.
        """
        # Walk the tree.
        for root, directories, files in os.walk(dataset_dir):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                if filename.find('_y.bin') > 0:  # get only the png files
                    filepath = os.path.join(root, filename)
                    self.file_test.append(filepath)  # Add it to the list.

    def write_csv(self):
        """
        Write train, test and validation.
        """

        file_name = dataset_dir +'/../train.csv'
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["id", "FileName"])
        file_id = 0
        for files in self.file_train:
            writer.writerow([file_id, files])
            file_id += 1
        csv_file.close()

        file_name = dataset_dir + '/../test.csv'
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["id", "FileName"])
        file_id = 0
        for files in self.file_test:
            writer.writerow([file_id, files])
            file_id += 1
        csv_file.close()

        file_name = dataset_dir + '/../val.csv'
        csv_file = open(file_name, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["id", "FileName"])
        file_id = 0
        for files in self.file_val:
            writer.writerow([file_id, files])
            file_id += 1
        csv_file.close()

    def run(self):
        self.get_test_list()
        self.get_train_val_list()
        self.write_csv()


data = PrepData()
data.run()
