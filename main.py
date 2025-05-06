import argparse
from preprocessing.preprocessing import Preprocessor
import utils
import os
from tqdm.auto import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', type=str, required=True)
    parser.add_argument('--dlib', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    preprocessor = Preprocessor(args.yolo, args.dlib)
    for i in tqdm(os.listdir(args.base_path)):
        img = preprocessor.preprocess(os.path.join(args.base_path, i))
        utils.save_img(img, os.path.join(args.save_path, i))

if __name__ == '__main__':
    main()