# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import ipdb
import scipy.io as sio

class TextData(data.Dataset):

    def __init__(self, root, image_path_txt):
        self.root = root

        with open(image_path_txt, 'r') as f:
            self.imgs_info = [line for line in f]       

        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def __getitem__(self, index):
        img_info = self.imgs_info[index]
        # true->1, false->0
        label = int(img_info.split()[-1])
        img_path = os.path.join(self.root, img_info.split()[0])
        data = Image.open(img_path)
        data = np.array(data)[:,:,:3]
        data = self.transform(Image.fromarray(data))
        return data, label
    
    def __len__(self):
        return len(self.imgs_info)

class Test_TextData(data.Dataset):

    def __init__(self, test_data_paths):
        
        self.imgs = [os.path.join(test_data_paths, img) for img in os.listdir(test_data_paths)]

        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def __getitem__(self, index):
        img = self.imgs[index]
        # true->1, false->0
        data = Image.open(img)
        data = np.array(data)[:,:,:3]
        data = self.transform(Image.fromarray(data))

        filename = img.split('/')[-1]
        return data, filename
    
    def __len__(self):
        return len(self.imgs)



