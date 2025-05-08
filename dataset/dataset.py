import mxnet
from mxnet import recordio
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from preprocessing.preprocessing import Preprocessor

class ImgDataset(Dataset):
    def __init__(self, rec_path, idx_path, yolo_path, dlib_path, transform=None):
        self.rec_path = rec_path
        self.idx_path = idx_path
        self.reader = recordio.MXIndexedRecordIO(self.idx_path, self.rec_path, 'r')
        self.keys = sorted(self.reader.keys)
        self.transform = transform
        self.preprocessor = Preprocessor(yolo_path, dlib_path)
        pass

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index + 1]
        record = self.reader.read_idx(key)

        header, img = recordio.unpack_img(record, iscolor=1)
        img = self.preprocessor.preprocess(img)
        img = Image.fromarray(img.astype(np.uint8))
        label = int(header.label)

        if self.transform:
            img = self.transform(img)
        
        return img, label
    
if __name__ == '__main__':
    print(mxnet.__version__)