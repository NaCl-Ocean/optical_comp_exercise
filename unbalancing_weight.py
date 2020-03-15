
import os,sys
import numpy as np
import cv2
import bs4 as bs
import glob
from PIL import Image

images_path = 'E:/02竞赛/水下目标检测/code//watert_comp_exercise/data/train/train/image'
gt_bbox_path = 'E:/02竞赛/水下目标检测/code//watert_comp_exercise/data/train/train/box'
image_name = os.listdir(images_path)
print(image_name)

n_background = 0
n_holothurian = 0
n_echinus = 0
n_scallop= 0
n_starfish = 0

#pixel_background = 0
pixel_holothurian = 0
pixel_echinus = 0
pixel_scallop = 0
pixel_starfish = 0
pixel_image = 0

def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size//2] + data[size//2-1])/2
        data[0] = median
    if size % 2 == 1:
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]



def count(img,gt_bbox_file):
    global pixel_holothurian, pixel_echinus, pixel_scallop, pixel_starfish, pixel_image
    with open(gt_bbox_file, 'r') as f:
        boxContent = f.read()
    soup = bs.BeautifulSoup(boxContent, features='xml')
    objects = soup.findAll("object")
    gt_boundingbox = np.zeros((len(objects), 5)).astype(np.int32)
    for i, object in enumerate(objects):
        if object.contents[1].string != 'waterweeds':
            object_area = (int(object.bndbox.xmax.getText())-int(object.bndbox.xmin.getText()))*\
                            (int(object.bndbox.ymax.getText())-int(object.bndbox.ymin.getText()))
            if object.contents[1].string == 'holothurian':
                pixel_holothurian += object_area
            if object.contents[1].string == 'echinus':
                pixel_echinus += object_area
            if object.contents[1].string == 'scallop':
                pixel_scallop += object_area
            if object.contents[1].string == 'starfish':
                pixel_starfish += object_area
    height,width,channels = img.shape
    pixel_image +=height*width
    return pixel_image,pixel_echinus,pixel_scallop,pixel_holothurian,pixel_starfish


image_files = glob.glob(images_path+'/*.jpg')
gt_bbox_files = glob.glob(gt_bbox_path+'/*.xml')
for image_file,gt_bbox_file in zip(image_files,gt_bbox_files):
    im = np.array(Image.open(image_file,'r'))
    height,width,channel = im.shape
    print(image_file )
    #print('before:',pixel_background)
    # print('before:',pixel_holothurian)
    # print('before:',pixel_echinus)
    # print('before:',pixel_scallop)
    # print('before:',pixel_starfish)

    #background_before = pixel_background
    holothurian_before = pixel_holothurian
    echinus_before = pixel_echinus
    scallop_before = pixel_scallop
    starfish_before = pixel_starfish

    count(im,gt_bbox_file)

    #if background_before - pixel_background != 0:
     #   n_background += 1
      #  print(n_background)
    if holothurian_before - pixel_holothurian != 0:
        n_holothurian += width*height
    if echinus_before - pixel_echinus != 0:
        n_echinus += width*height
    if scallop_before - pixel_scallop != 0:
        n_scallop += width*height
    if starfish_before - pixel_starfish != 0:
        n_starfish += width*height



pixel_background = pixel_image - pixel_starfish - pixel_starfish - pixel_echinus - pixel_holothurian

f_holothurian = pixel_holothurian/(n_holothurian)
f_echinus = pixel_echinus/(n_echinus)
f_scallop = pixel_scallop/(n_scallop)
f_starfish = pixel_starfish/(n_starfish)
f_background = pixel_background/(pixel_image)

median_f = [f_holothurian,f_echinus,f_scallop,f_starfish,f_background]

#weight(class) = median of f(class)) / f(class)
median = get_median(median_f)
#weight_background = median/f_background
weight_holothurian = median/f_holothurian
weight_echinus = median/f_echinus
weight_scallop = median/f_scallop
weight_starfish = median/f_starfish
weight_background = median/f_background


#print('weight_background:',weight_background)
print('weight_echinus:',weight_echinus)
print('weight_holothurian:',weight_holothurian)
print('weight_scallop:',weight_scallop)
print('weight_starfish:',weight_starfish)
print('weight_background:',weight_background)