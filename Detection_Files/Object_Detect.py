import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import sys

from detectme import *
from detectron2.utils.visualizer import Visualizer

detector = Detector(model_type= "IS")

def plotfun(image_path):

    image_out, mask_out, index_finals, class_num, scores_finals = detector.onImage(image_path)  # you should make a dataloader of one image
    scores_finals_FVersion = scores_finals.tolist()

    #computed_image=detector.plot_det(image_path)

    return image_out, mask_out, class_num 

def plot_det_object(image_path):
    
    plot_out=detector.plot_det(image_path)
    
    return plot_out


