from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists():
    image_dir='/home/rick/derma/dataset'
    testing_percentage=20


    result = {}
    counter_for_result_label=0

    sub_dirs = [x[0] for x in gfile.Walk(image_dir)] #create sub_dirs


    # The root directory comes first, so skip it.

    dir_name=[]

    #ignore first element in sub_dir
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue


        dir_name = os.path.basename(sub_dir)

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []

        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
          #for image_dir in sub_dir
          file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
          file_list.extend(gfile.Glob(file_glob))      #create a list of all files


        #using regex to set label name
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        #dividing
        training_images = []
        testing_images = []
        for file_name in file_list:
          base_name = os.path.basename(file_name) #just take name of image (eg: 5547758_ed54_n)

          hash_name = re.sub(r'_nohash_.*$', '', file_name)

          hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
          percentage_hash = ((int(hash_name_hashed, 16) %
                              (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
          if percentage_hash < testing_percentage:
            #testing_images.append(file_name)
            testing_images.append(cv2.imread(file_name))
            #testing_images.append(base_name)           
          else:
            #training_images.append(file_name)
            training_images.append(cv2.imread(file_name))
            #training_images.append(base_name)


        result[counter_for_result_label] = {
        'training_label': [counter_for_result_label]*(len(training_images)),
        'testing_label': [counter_for_result_label]*(len(testing_images)),
        'training': training_images,
        'testing': testing_images,
        }

        counter_for_result_label=counter_for_result_label + 1

    return result
