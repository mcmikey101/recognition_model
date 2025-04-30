import cv2
import numpy as np

def save_img(image, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(np.array(image).squeeze(0), cv2.COLOR_BGR2RGB))