import bs4 as bs
from PIL import Image
import cv2.cv2
from __init__ import *
import os

class_dict ={'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}

def get_gt_box(gt_box_file):
    with open(gt_box_file, 'r') as f:
        boxContent = f.read()
    soup = bs.BeautifulSoup(boxContent, features='xml')
    objects = soup.findAll("object")
    gt_boundingbox = np.zeros((len(objects), 5)).astype(np.int32)
    for i, object in enumerate(objects):
        gt_boundingbox[i, 1] = int(object.bndbox.xmin.getText())
        gt_boundingbox[i, 2] = int(object.bndbox.ymin.getText())
        gt_boundingbox[i, 3] = int(object.bndbox.xmax.getText())
        gt_boundingbox[i, 4] = int(object.bndbox.ymax.getText())
        gt_boundingbox[i, 0] = class_dict[object.contents[1].string]
    return gt_boundingbox


def get_label(gt_box_file, height, width):
    label = np.ones((height, width, 1)).astype((np.int32)) * 255
    gt_bbox = get_gt_box(gt_box_file)
    for i in range(gt_bbox.shape[0]):
        label[gt_bbox[i, 2]:gt_bbox[i, 4], gt_bbox[i, 1]:gt_bbox[i, 3], 0] = gt_bbox[i, 0]
    return label

def get_image(image_file):
    image = np.array(Image.open(image_file))
    height, width, channels = image.shape
    image = image.astype((np.int32))
    return image, height, width

main_box_dir = 'E:/02竞赛/水下目标检测/water_optical_comp/train/train/box/'
main_image_dir = 'E:/02竞赛/水下目标检测/water_optical_comp/train/train/image/'

for i in range(1,1000):
    image_file = main_image_dir + str("%06d" % i) + '.jpg'
    gt_box_file = main_box_dir + str("%06d" %  i) + '.xml'
    image, height, width = get_image(image_file)
    label = get_label(gt_box_file, height, width) * 60
    label = np.concatenate((label, label,label),axis=-1).astype(np.uint8)
    image = image.astype(np.uint8)
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 640, 480)
    cv2.imshow("image", image)
    cv2.namedWindow("label", 0)
    cv2.resizeWindow("label", 640, 480)
    cv2.imshow('label' ,label)
    cv2.waitKey(0)


