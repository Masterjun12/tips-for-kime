import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from pathlib import Path
import math
from torch.utils.data import Subset

'''
torchvision에서의 사용가능한 일반적인 데이터셋 중 하나는 ``ImageFolder`` 입니다.
이것은 다음과 같은 방식으로 구성되어 있다고 가정합니다: ::

    root/ants/xxx.png
    root/ants/xxy.jpeg
    root/ants/xxz.png
    .
    .
    .
    root/bees/123.jpg
    root/bees/nsdf3.png
    root/bees/asd932_.png

여기서'ants', 'bees'는 class labels입니다.
'''

des_path = Path(__file__).parents[1].joinpath('data/ND2-Neuron/')
#des_path = os.path.join(os.getcwd(),'dataset\\class10\\images')

def get_data_loader(data_dir= des_path, batch_size=4, num_workers=0):
    """
    Define the way we compose the batch dataset including the augmentation for increasing the number of data
    and return the augmented batch-dataset
    :param data_dir: root directory where the either train or test dataset is
    :param batch_size: size of the batch
    :param train: true if current phase is training, else false
    :return: augmented batch dataset
    """

    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'aug_train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),        
        'val': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),        
        'test': transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    load_setnames = ['aug_train', 'val', 'test']

    # ImageFloder with root directory and defined transformation methods for batch as well as data augmentation    
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), transform[x])
                for x in load_setnames}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                for x in load_setnames}

    return dataloaders




###################################################
# Main
###################################################
# The way to get one batch from the data_loader
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    data_loaders = get_data_loader()

    for i in range(10):
        batch_x, batch_y = next(iter(data_loaders['test']))
        print(np.shape(batch_x), batch_y)