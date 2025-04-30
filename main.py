import argparse
from preprocessing.preprocessing import Preprocessor
import utils.utils as utils
import datasets.datasets as datasets
import os
import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', required=True)
    parser.add_argument('--dlib', required=True)
    parser.add_argument('--images_path', required=True)
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preprocessor = Preprocessor(args.yolo, args.dlib)
    tensor = preprocessor.preprocess(args.images_path)
    utils.save_tensor(tensor, args.save_path)

if __name__ == '__main__':
    main()