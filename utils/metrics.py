from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

def getTARFAR(true, pred):
    far, tar, threshold = roc_curve(true, pred)
    far_percent = np.round(far * 100, 2)
    tar_percent = np.round(tar * 100, 2)
    print('TAR@FAR=1%: ', tar_percent[np.where(np.isclose(far_percent, 1))[0][0]])
    print('Threshold for TAR@FAR=1%: ', threshold[np.where(np.isclose(far_percent, 1))[0][0]])