import argparse
from preprocessing.preprocessing import Preprocessor
import utils
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_path', type=str, required=True)
    parser.add_argument('--dlib_path', type=str, required=True)
    parser.add_argument('--rec_path', type=str, required=True)
    parser.add_argument('--idx_path', type=str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--proc_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.base_path, exist_ok=True)
    os.makedirs(args.proc_path, exist_ok=True)

    utils.write_rec_file(args.rec_path, args.idx_path, args.base_path)

    preprocessor = Preprocessor(args.yolo_path, args.dlib_path)
    c = 0
    for i in sorted(os.listdir(args.base_path)):
        tensor = preprocessor.preprocess(args.base_path + "/" + i)
        for face in tensor:
            utils.save_img(face, args.proc_path + "/" + f'image_{c}.jpg')
        c += 1

if __name__ == '__main__':
    main()