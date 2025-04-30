import argparse
from preprocessing.preprocessing import Preprocessor
import utils
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True)
    parser.add_argument('--dlib_path', type=str, required=True)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    preprocessor = Preprocessor(args.yolo_path, args.dlib_path)
    for i in os.listdir(args.images_path):
        tensor = preprocessor.preprocess(os.path.join(args.images_path, i))
        utils.save_img(tensor, os.path.join(args.save_path, i))

if __name__ == '__main__':
    main()