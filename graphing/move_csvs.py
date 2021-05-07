import os
import glob
from distutils.dir_util import copy_tree

from_directory = "D:/Desktop/kxr758/logs/final_final/regularisation/l2"

to_directory = "D:/Desktop/graphs/new system/regularisation/all-100-class weights/l2"
logs_code = ["20560810",
             "01003010",
             "00520910",
             "00430710",
             "20491010",
             "20421410",
             ]

labels = [str(1 * 10 ** -i) for i in range(1, 7)]

for dates in os.listdir(from_directory):
    dates_folder = os.path.join(from_directory, dates)
    for runs in os.listdir(dates_folder):
        runs_path = os.path.join(dates_folder, runs).replace("\\", "/")

        for idx, code in enumerate(logs_code):
            if code in str(runs_path):
                # copy
                # get train and validation folder
                train_path = os.path.join(runs_path, 'train').replace("\\", "/")
                validation_path = os.path.join(runs_path, 'validation').replace("\\", "/")
                copy_tree(train_path, os.path.join(to_directory, labels[idx], 'train'))
                copy_tree(validation_path, os.path.join(to_directory, labels[idx], 'validation'))
