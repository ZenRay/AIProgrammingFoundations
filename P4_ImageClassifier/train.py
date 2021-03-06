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

        # freeze the model grad
        for param in self.model.parameters():
            param.requires_grad = False

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

    def __test_report(self, loader):
        """Display the test report
        """
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            
            # change the model stage
            self.model.eval()

            test_loss = 0
            accuracy = 0

            for images, labels in iter(loader):
                if torch.cuda.is_available():
                    inputs = Variable(images.float().cuda(), volatile=True)
                    labels = Variable(labels.long().cuda(), volatile=True)
                else:
                    inputs = Variable(images, volatile=True)
                    labels = Variable(labels, volatile=True)

                output = self.model.forward(inputs)
                test_loss += self.criterion(output, labels).data[0]

                # calculate the probability
                ps = torch.exp(output).data
                # class with highest probability, compared with true label
                equality = (labels.data == ps.max(1)[1])

                # accuracy is correct predictions ratio
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            return test_loss/len(loader), accuracy/len(loader) * 100

    def __loader(self, rotation, resize):
        """
        Get the dataset loader
        """
        # define the transforms
        train_transforms = transforms.Compose([
            transforms.RandomRotation(rotation),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomResizedCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
            
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])

        # load the datasets by define the dataloaders
        self.train_datasets = datasets.ImageFolder(
            self.train, transform=train_transforms
        )
        self.valid_datasets = datasets.ImageFolder(
            self.validate, transform=test_transforms
        )
        self.test_dataset = datasets.ImageFolder(
            self.test, transform=test_transforms
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_datasets, batch_size=64, shuffle=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_datasets, batch_size=32
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32
        )
        return self.train_loader, self.valid_loader, self.test_loader

    def train_model(
        self, input_size, out_size, rotation, resize, hidden_layers=None, epochs=2
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
        epochs: int
            The train epoch
        Returns:
        ----------
        model:
            Nerual Network model
        criterion:
            Nerual Network criterion
        optimizer:
            Nerual Network optimizer
        """

        self.__classifier(input_size, out_size, hidden_layers)

        # get the dataset loader
        self.__loader(rotation, resize)

        # define criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

        # push the model into gpu
        self.model.cuda()
        self.model.to("cuda")
        # change the model stage
        self.model.train()

        print_steps = int(len(self.train_loader) / 4)
        train_loss = 0
        steps = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # train the network
            for epoch in range(0, epochs):
                print("Epoch: {}/{}\n{}".format(epoch+1, epochs, "_"*20))
                for inputs, labels in iter(self.train_loader):
                    steps += 1
                
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                    self.optimizer.zero_grad()

                    ouputs = self.model.forward(inputs)
                    loss = self.criterion(ouputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    # display the report
                    if steps % print_steps == 0:
                        print(
                            "\tTrain Loss:\t{:.4f}".format(train_loss / print_steps),
                            "Validation Loss:\t{:.4f}\tValidation Accuracy:\t{:.4f}".format(
                                *self.__test_report(self.valid_loader)
                            )
                        )
                        # refix tthe loss value
                        train_loss = 0
        return self.model, self.criterion, self.optimizer

    def dump_checkpoint(
        self, input_size, output_size, hidden_layers=None, 
        name="vgg19bn_checkpoint.pth"
    ):
        """
        Save the checkpoint

        Parameters:
        -----------
        input_size: int
            Nerual Network input size
        out_size: int
            Nerual Network output size
        hidden_laysers: list default None
            Nerual Network hidden layers
        name: string
            Checkpoint file name
        """
        class_to_idx = {
            val: key for key, val in self.train_datasets.class_to_idx.items()
        }

        checkpoint = {
            "input_size": input_size,
            "output_size": output_size,
            "class_to_idx": class_to_idx,
            "state_dict": self.model.state_dict(),
            "hidden_layers": hidden_layers
        }

        torch.save(checkpoint, name)