import Networks
import h5py
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
import paddle.vision.transforms as TF
from paddle.nn.initializer import Assign, Normal, Constant
import random
import cv2

import matplotlib.pyplot as plt
%matplotlib inline
import time

model_path = './output/'

reader = data_reader(cfg)

def adaIN_trainer(x1, x2, adaIN, adaIN_optimizer):

    y, lc, ls = adaIN(x1, x2)
    #lide = adaIN.mse_loss(x1, y)

    loss = lc + 10*ls# + lide

    adaIN_optimizer.clear_grad()
    loss.backward()
    adaIN_optimizer.minimize(loss)

    return loss.numpy()

