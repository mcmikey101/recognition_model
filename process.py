from mxnet import recordio
import cv2
from preprocessing.aligner import Aligner
import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path='./configs', config_name='preprocessing')
def main(cfg: DictConfig):
    os.makedirs(cfg.save_path, exist_ok=True)

    aligner = Aligner(cfg.yolo_path, cfg.dlib_path)
    reader = recordio.MXIndexedRecordIO(cfg.idx_path, cfg.rec_path, 'r')
    keys = sorted(reader.keys)

    for i in range(1, len(keys)):
        key = keys[i]
        record = reader.read_idx(key)

        header, img = recordio.unpack_img(record, iscolor=1)
        img = aligner(img)
        label = int(header.label)
        cv2.imwrite(f"{os.path.join(cfg.save_path, str(label) + '_' + str(i))}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    main()