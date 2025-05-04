import cv2
import numpy as np
import mxnet as mx

def save_img(image, output_path):
    cv2.imwrite(output_path, np.array(image))

def read_rec(rec_path, idx_path):
    data_iter = mx.image.ImageIter(
        batch_size=1,
        data_shape=(3, 112, 112),
        path_imgrec=rec_path,
        path_imgidx=idx_path,
    )
    return data_iter
    