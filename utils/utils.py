import cv2
import numpy as np
import mxnet as mx
import os

def save_img(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))

def write_rec_file(rec_path, idx_path, output_path):
    data_iter = mx.image.ImageIter(
        batch_size=1,
        data_shape=(3, 112, 112),
        path_imgrec=rec_path,
        path_imgidx=idx_path,
    )
    data_iter.reset()
    for i in range(10):
        batch = data_iter.next()
        data = batch.data[0].asnumpy().astype(np.uint8).squeeze(0).transpose((1, 2, 0))
        save_img(data, output_path + f"/image_{i}.jpg")