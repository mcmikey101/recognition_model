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
    parser.add_argument('--processed_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.processed_path, exist_ok=True)

    base_imgs = utils.read_rec(args.rec_path, args.idx_path)

    preprocessor = Preprocessor(args.yolo_path, args.dlib_path)
    c = 0
    for i in base_imgs:
        batch = base_imgs.next()
        data = batch.data[0].asnumpy().astype(np.uint8).squeeze(0).transpose((1, 2, 0))
        data = preprocessor.preprocess(data)
        for face in data:
            utils.save_img(face, args.processed_path + "/" + f'image_{c}.jpg')
        c += 1

if __name__ == '__main__':
    main()