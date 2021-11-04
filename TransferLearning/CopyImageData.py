import os
from shutil import copy2, rmtree
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor
from PIL import Image
import numpy as np

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
# class bcolors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'

# print(f"{bcolors.WARNING}Error : Test message !{bcolors.ENDC}")
def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def Copy_Neuron_NoNeuron_Images_forImageFolder(org_path, des_path):
    
    Labels = ['Neuron', 'NoNeuron']
    Datasets = ['train', 'val', 'test']

    # Clean up Destination Folder
    #rmtree(des_path, ignore_errors=True)

    # Create Destination Label Folders [Stage 0]
    for dataset in Datasets:
        for lbl in Labels:
            try:
                os.makedirs(os.path.join(des_path, dataset, lbl))
            except:
                pass

    # Scan root dir for label & path from Orginal Folder [Stage 1]
    LabeledFolders = []
    for entry in os.scandir(org_path):
        if entry.is_dir():
            dirname = os.path.basename(entry.name)
            if dirname.startswith('[Neuron]'):
                LabeledFolders.append(('Neuron', 0, entry.path))
            elif dirname.startswith('[NoNeuron]'):
                LabeledFolders.append(('NoNeuron', 1, entry.path))


    # Update the filename, label [Stage 2]    
    fn_lbl = {"Neuron":0, "NoNeuron":0}
    #fn_totla = sum(fn_lbl.values())
    fn_info = []
    for labelname, label, folder in LabeledFolders:
        for base, dirs, files in os.walk(folder):
            dirname = os.path.basename(base)
            datasetname = os.path.basename(folder)
            
            if dirname.startswith('back'):
                continue

            for f in files:
                if f.endswith(('.tif', '.tiff', '.png')):                    
                    fn_lbl[labelname] += 1
                    fn_info.append( (label, labelname, fn_lbl[labelname], base, f, datasetname) )

    """
    Seperate Train/Val/Test
    """
    # StratifiedShuffleSplit sample the data in same proportion of labels
    indices = range(len(fn_info))
    y_ = [x[1] for x in fn_info]

    # Test(20%) & OrgTrain(80%) Dataset
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_val_index, test_index in split1.split(indices, y_):
        print('test:', test_index)
        print('train_val(len):', len(train_val_index), 'test(len):', len(test_index))
    
    # @OrgTrain(80%) -> Train(70%) & Val(30%) Dataset
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, val_index in split2.split(train_val_index, np.array(y_)[train_val_index.astype(int)]):
        print('train:', train_index)
        print('val:', val_index)
        print('train(len):', len(train_index), 'val(len):', len(val_index), 'test(len):', len(test_index))


    # Copy image files to destination folder w/ label [Stage 2]
    datasets_idx = [train_index, val_index, test_index]
    for i, dataset in enumerate(Datasets):
        for train_idx in datasets_idx[i]:
            labelname = fn_info[train_idx][1]
            label_i = fn_info[train_idx][2]
            base = fn_info[train_idx][3]
            f = fn_info[train_idx][4]
            datasetname = fn_info[train_idx][5]
            
            src_file = os.path.join(base, f)
            dst_file = os.path.join(des_path, dataset, labelname, datasetname + "_" + f)
            print("[{0}][{1}-{2}] - Copy {3} to {4}".format(train_idx, labelname, label_i, src_file, dst_file))

            if os.path.exists(dst_file):  
                if os.path.samefile(src_file, dst_file):
                    continue
                else:                    
                    filename, file_extension = os.path.splitext(f)
                    dst_file = os.path.join(des_path, dataset, labelname, filename+"_copy"+file_extension)
                    print(colored(255,0,0,"[Overwrite] [{0}][{1}-{2}] - Copy {3} to {4}".format(train_idx, labelname, label_i, src_file, dst_file)))
                    #raise Exception('Try to copy same file.')
                    #raise Warning('Try to copy same file.')

            copy2(src_file, dst_file)




def Augmentation_TrainingImages(org_path, Folder_Name):
    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.RandomCrop(256),
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
        ]),
    }

    # ImageFloder with root directory and defined transformation methods for batch as well as data augmentation    
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(org_path, x), transform[x])
                for x in ['train']}


    # Make label folders under the FolderName
    for name in image_datasets['train'].classes:
        try:
            os.makedirs(os.path.join(org_path, Folder_Name, name))
        except:
            pass

    for rep in range(3):
        for num, value in enumerate(image_datasets['train']):
            data, label_idx = value
            label = image_datasets['train'].classes[label_idx]
            data.save(os.path.join(org_path, Folder_Name, label, '%d_%d_%d.jpg'%(rep, num, label_idx)))




###################################################
# Main
###################################################
# The way to get one batch from the data_loader
if __name__ == "__main__":
    org_path = r'G:\내 드라이브\Datasets\NeuroImage\ND2-NeuronVsNon'
    des_path = Path(__file__).parents[1].joinpath('data/ND2-Neuron/')

    Copy_Neuron_NoNeuron_Images_forImageFolder(org_path, des_path)
    Augmentation_TrainingImages(des_path, 'aug_train')