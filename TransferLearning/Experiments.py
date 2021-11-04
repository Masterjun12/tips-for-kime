#!/usr/bin/env python3

# from __future__ import print_function 
# from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from TrainModels import *

###################################################
# Main
###################################################
# The way to get one batch from the data_loader
from LoadData import get_data_loader
def GetUpperDir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),".."))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support() #Window 환경에서의 RuntimeError 방지 (DataLoader의 num_workers 관련)

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define argparser
    parser = argparse.ArgumentParser(description='NeuroImage_Neuron')
    args, unknown = parser.parse_known_args()


    """
    Experiment Parameters
    """
    # Top level data directory. Here we assume the format of the directory conforms 
    #   to the ImageFolder structure
    #args.data_dir = os.path.join(GetUpperDir(), "data/hymenoptera_data")

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    args.model_name = "squeezenet"

    # Number of classes in the dataset
    args.num_classes = 2

    # Batch size for training (change depending on how much memory you have)
    args.batch_size = 64

    # Number of epochs to train for 
    args.num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model, 
    #   when True we only update the reshaped layer params
    args.feature_extract = True

    """
    Initialize the Model for Transfer Learning
    """
    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)


    """
    Load Data
    """
    # From Data Folders
    print("Initializing Datasets and Dataloaders...")
    data_loaders = get_data_loader()


    """
    Create the Optimizer
    """
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()


    """"
    Run Training and Validation Step
    """
    # Train and evaluate
    model_ft, hist = train_model(model_ft, data_loaders, criterion, optimizer_ft, 
        device, 'aug_train', 'val', num_epochs=args.num_epochs, is_inception=(args.model_name=="inception"))


    # Plot the training curves of validation accuracy vs. number 
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = [h.cpu().numpy() for h in hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,args.num_epochs+1),ohist,label="Pretrained")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, args.num_epochs+1, 1.0))
    plt.legend()
    plt.show(block=True)
