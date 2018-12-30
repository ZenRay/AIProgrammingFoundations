#!/usr/bin/env python
# -*-coding: utf-8 -*-

"""Train Neural Network
    Train the Neural Network, by using the file.
"""

import argparse
import warnings
import os
import json

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
        # set the default models
        pretrain_model = {
            "vgg19_bn": models.vgg19_bn(pretrained=True), 
            "vgg19": models.vgg19(pretrained=True),
            "resnet50": models.resnet50(pretrained=True)
        }

        self.dir = dir
        self.train = os.path.join(dir, "train")
        self.test = os.path.join(dir, "test")
        self. validate = os.path.join(dir, "valid")
        
        self.file = file
        try:
            self.model = pretrain_model[arch]
        except KeyError:
            arch = input("Invalidate model, choose vgg19_nb, vgg19, resnet50: ")
            self.model = pretrain_model[arch]

    def get_spacies(self, file=None):
        """Parse the flowers spacies
        Parse the json file store the flower spacies and category
        """
        if file is None:
            current_file = self.file
        else:
            current_file = file
        
        with open(current_file, "r") as data:
            self.spacies = json.load(data)
        
        return self.spacies

    def __classifier(self, input_size, out_size, hidden_layers=None):
        """Build the classifier

        Enter your Nerual Network layers information
        
        Parameters:
        -----------
        input_size: int
            Nerual Network input size
        out_size: int
            Nerual Network output size
        hidden_laysers: list default None
            Nerual Network hidden layers
        """
        if not isinstance(hidden_layers, list):
            hidden_layers = [1000, 300]
        elif len(hidden_layers) > 2:
            hidden_layers = [1000, 300]
        
        classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size, hidden_layers[0])),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(hidden_layers[0], hidden_layers[1])),
            ("relu2", nn.ReLU()),
            ("dropout", nn.Dropout(p=.5)),
            ("fc3", nn.Linear(hidden_layers[1], out_size)),
            ("output", nn.LogSoftmax(dim=1))
        ]))

        # adjust the classifier
        self.model.classifier = classifier
        return self.model

    def train_model(
        self, input_size, out_size, rotation, reseize, hidden_layers=None
    ):
        """Build the classifier

        Enter your Nerual Network layers information
        
        Parameters:
        -----------
        input_size: int
            Nerual Network input size
        out_size: int
            Nerual Network output size
        hidden_laysers: list default None
            Nerual Network hidden layers
        rotation: int
            Transforms Random Rotations degree
        resize: int
            The final image size
        """

        self.__classifier(input_size, out_size, hidden_layers)

        # define the transforms
        train_transforms = transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomResizedCrop(reseize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
            
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(reseize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])

        # load the datasets by define the dataloaders
        train_datasets = datasets.ImageFolder(
            self.train, transform=train_transforms
        )
        valid_datasets = datasets.ImageFolder(
            self.validate, transform=test_transforms
        )
        test_dataset = datasets.ImageFolder(
            self.test, transform=test_transforms
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=64, shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_datasets, batch_size=32
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32
        )

        # define criterion and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

        # push the model into gpu
        self.model.cuda()

        return self.model