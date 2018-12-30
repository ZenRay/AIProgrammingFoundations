#!/usr/bin/env python
# -*-coding: utf-8 -*-

"""Train Neural Network
    Train the Neural Network, by using the file.
"""

import argparse
import warnings
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
from PIL import Image

# load torch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value '/data/aipnd_projects/flowers'
      2. CNN Model Architecture as --arch with default value 'vgg19_bn'
      3. Text File with Spacies Names as --dogfile with default value 'dognames.txt'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
   
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--dir", type=str, default="/data/aipnd_projects/flowers",
        help="Image Folder default value flowers"
    )

    # pretrained model
    parser.add_argument(
        "-a", "--arch", type=str, default="vgg19_bn",
        help="Model archtecture default value 'vgg19_bn"
    )

    # spacies name 
    parser.add_argument(
        "-f", "--file", type=str, default="cat_to_name.json"
    )

    # get the argments
    args = parser.parse_args()

    return args

class TrainModel:
    def __init__(self, dir, arch, file):
        """Init the variable

        Parameters:
        -----------
        dir: string
            Image path
        arch: string
            CNN model archtecture
        file: json file
            Text file with spacies name
        """
        self.dir = dir
        self.train = os.path.join(dir, "train")
        self.test = os.path.join(dir, "test")
        self. validate = os.path.join(dir, "valid")
        
        self.file = file
        self.arch = arch

        