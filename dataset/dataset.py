from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import cv2

class ImgDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.images = sorted(os.listdir(images_path))
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.images_path, self.images[index]))
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = sorted(os.listdir(self.images_path))[index].split('_')[0]
        return img, label
    