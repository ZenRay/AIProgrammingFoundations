#!/usr/bin/env python
#-*-coding: utf-8-*-

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
      4. Checkpoint file with model as --model with default value 'vgg19bn_checkpoint.pth'
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

    # model checkpoint
    parser.add_argument(
        "-m", "--model", type=str, default="vgg19bn_checkpoint.pth",
        help="Model checkpoint file"
    )

    # get the argments
    args = parser.parse_args()

    return args

def get_spacies(file):
    """Parse the flowers spacies
    Parse the json file store the flower spacies and category

    Parameters:
    -----------
    file: string
        Category name file
    
    Returns:
    ----------
    result: dict
        Key value about the category name and flower name
    """
    with open(file, "r") as data:
        result = json.load(data)
    
    return result

def load_checkpoint(filepath):
    """Load the model checkpoint
    Parameters:
    -----------
    filepath: string
    Load the model check point
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # load the checkpoint
        checkpoint = torch.load(filepath)
        # init the model
        model = models.vgg19_bn(pretrained=True)
        #freeze the parameters of the pre-trained model.
        for param in model.parameters():
            param.requires_grad = False


        #build the classifier with input, hidden, and output units.
        classifier = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"][0])), 
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(checkpoint["hidden_layers"][0], checkpoint["hidden_layers"][1])),
            ("relu2", nn.ReLU()),
            ("dropout", nn.Dropout(p=0.5)),
            ("fc3", nn.Linear(checkpoint["hidden_layers"][1], checkpoint["output_size"])),
            ("output", nn.LogSoftmax(dim=1))

        ]))

        # set the classifier
        model.classifier = classifier

        model.load_state_dict(checkpoint["state_dict"])
        print("class_to_idx", checkpoint["class_to_idx"])
    #     train_datasets.class_to_idx = checkpoint["class_to_idx"]
        return model, checkpoint["class_to_idx"]

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: Process a PIL image for use in a PyTorch model
        image = Image.open(image)

        # get the width and height
        width, height = image.size

        # set the new image
        if width > height:
            width = int(256 / height * width)
            image = image.resize((width, 256), Image.ANTIALIAS)
        else:
            height = int(256 / width * height)
            image = image.resize((256, height), Image.ANTIALIAS)

        # get the new width and new height
        width, height = image.size

        # crop the image from the center
        left = (width - 224) / 2
        right = (width + 224) / 2
        up = (height - 224) / 2
        bottom = (height + 224) / 2

        # scale the image 
        image = image.crop(box=(left, up, right, bottom))

        # get the iamge data
        data = np.array(image) / 255

        # normalize the data
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        data = (data - mean) / std

        # transpose the data
        data = np.transpose(data, (2, 0, 1))

        return data

def imshow(image, ax=None, title=None, show_option=False):
    """Imshow for Tensor.
    Parameters:
    -----------
    image: torch data
        torch data
    ax: Axes default None 
        A Figure axes
    title: string
        The flower name
    show_option: boolean default False
        If it is True, show the iamge
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if show_option:
            if ax is None:
                _, ax = plt.subplots()

            # PyTorch tensors assume the color channel is the first dimension
            # but matplotlib assumes is the third dimension
            image = image.transpose((1, 2, 0))

            # Undo preprocessing
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean

            # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
            image = np.clip(image, 0, 1)

            ax.imshow(image)

            return ax

def predict(image_path, model, topk=5, class_to_idx=None):
    """ Predict the class (or classes) of an image using a trained deep learning model.

    Parameters:
    -----------
    image_path: string
        A flower path
    model: 
        Model object
    topk: int
        Top k category, which is predicted by the model
    class_to_idx: dict default None
        Checkpoint class_to_idx value
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.to("cuda")
        model.eval()
        img_tensor = torch.from_numpy(process_image(image_path))
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.type(torch.cuda.FloatTensor)
        output = model(Variable(img_tensor.cuda(), volatile=True))
        ps = torch.exp(output)
        probs, index = ps.topk(topk)
        probs = probs.cpu().detach().numpy().tolist()[0]
        index = index.cpu().detach().numpy().tolist()[0]

        index = [class_to_idx[i] for i in index]
        return probs, index

def show_result(
    path, classname, model, topk, cat_to_name=None, show_option=False,
    class_to_idx=None
):
    """Show a flower predict result
    
    Parameters:
    -----------
    path: string
        A flower path
    classname: string
        The flower category
    model: 
        Model object
    topk: int
        Top k category, which is predicted by the model
    cat_to_name:
        All flowers category name
    show_option: boolean default False
        If it is True, show the iamge
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots(nrows=2, figsize=(8, 16))

        # get the species
        species = cat_to_name[classname]

        # plot the flower
        data = process_image(path)
        imshow(data, ax[0], species, show_option)
        ax[0].set_title(species)
        # get the prediction probs
        probs, index = predict(path, model, class_to_idx=class_to_idx)
        index = [cat_to_name[i] for i in index]

        prediction = pd.Series(data=probs, index=index)
        prediction.plot(kind="barh", ax=ax[1])
        
        plt.show()

if __name__ == "__main__":
    args = get_input_args()

    # get cat_to_name
    cat_to_name = get_spacies(args.file)

    # load model
    model, class_to_idx = load_checkpoint(args.model)

    # predict report
    probs, classes = predict(os.path.join(args.dir, "train/1/image_06735.jpg"),
                        model, class_to_idx=class_to_idx
                )
    print("The model train result:")
    for prob, cat in zip(probs, classes):
        print("Name:  {:30s}\tProbability:{:10.4f}".format(cat_to_name[cat], prob))

    # show the test report
    show_result(os.path.join(args.dir, "test/101/image_07949.jpg"),
        "101", model, 5, cat_to_name=cat_to_name, show_option=True,
                class_to_idx=class_to_idx
    )
    # test result
    probs, classes = predict(os.path.join(args.dir, "test/101/image_07949.jpg"),
                        model, class_to_idx=class_to_idx)
    print("The model test result about {}:".format(cat_to_name["101"]))
    for prob, cat in zip(probs, classes):
        print("Name:  {:30s}\tProbability:{:10.4f}".format(cat_to_name[cat], prob))