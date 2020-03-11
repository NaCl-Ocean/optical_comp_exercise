import torch
import numpy as np
import Map_metric
import dataset
import unet_model
import train
import csv

class_dict ={'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}
class_dict_reverse = {1:'holothurian', 2:'echinus', 3:'scallop', 4:'starfish'}