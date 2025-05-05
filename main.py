import argparse
from datasets.datasets import ImgDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True)
    parser.add_argument('--dlib_path', type=str, required=True)
    parser.add_argument('--rec_path', type=str, required=True)
    parser.add_argument('--idx_path', type=str, required=True)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImgDataset(args.rec_path, args.idx_path, args.yolo_path, args.dlib_path, transform=transform)

if __name__ == '__main__':
    main()
