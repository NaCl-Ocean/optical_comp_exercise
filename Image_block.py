import os
import torch
import numpy as np
from torchvision import transforms as transforms
from PIL import Image
import glob

class Image_Block():
    def __init__(self,Image_dir,save_dir):
        self.Image_dir = Image_dir
        self.image_files = glob.glob(self.Image_dir+'/*.jpg')
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir): os.mkdir(self.save_dir)
    def get_image(self,i):
        image_path = self.image_files[i]
        image = np.array(Image.open(image_path,'r'))
        return image

    def Image_Blocking(self,image):
        '''
        Block images for  resolution (1920,1080) and (3840,2160)
        Resize image to Block image resoltuion (640,360)
        '''
        height, width, channel = image.shape
        if height==1080 and width == 1920:
            image_block = np.zeros((9,360,640,3)).astype(np.uint8)
            # image_block排列规则：[col1_row1,col1_row2,col1_row3,col2_row1,col2_row2,col2_row3,col3_row1,col3_row2,col3_row3]
            for col in range(3):
                for row in range(3):
                    image_block[row+col*3] = image[col*360:(col+1)*360, row*640:(row+1)*640, :]
        elif  height== 2160 and width == 3840:
            image_block = np.zeros((36, 360, 640, 3)).astype(np.uint8)
            for col in range(6):
                for row in range(6):
                    image_block[row + col*6] = image[col*360:(col + 1)*360, row*640:(row + 1)*640, :]
        else:
            transform_list = [transforms.Resize((360,640))]
            transform = transforms.Compose(transform_list)
            image = Image.fromarray(image)
            image_block = transform(image)
            image_block = np.array(image_block)
        return image_block


    def write_image(self,image,image_block,i):
        height, width, channel = image.shape
        image_id = get_image_id(self.image_files[i],os_name='windows')
        save_name = str("%06d" % image_id)+ '_'+ str(height) + '_'+str(width)
        if image_block.ndim == 4:
            for i in range(np.size(image_block,0)):
                image = image_block[i]
                image = Image.fromarray(image)
                save_path = save_name + '_' + str(i)
                save_path = os.path.join(self.save_dir, save_path)
                save_path = save_path.replace('\\', '/')
                image.save(save_path+'.jpg')
        else:
            image = Image.fromarray(image_block)
            save_path = save_name
            save_path = os.path.join(self.save_dir, save_path)
            save_path = save_path.replace('\\', '/')
            image.save(save_path+'.jpg')

    def Block_image_and_save(self):
        for i in range(len(self.image_files)):
            image = self.get_image(i)
            image_block = self.Image_Blocking(image)
            self.write_image(image,image_block,i)


def get_image_id(image_file,os_name='linux'):
    if os_name == 'linux':
        temp = image_file.split('/')
        image_id = int(temp[len(temp) - 1][:6])
    elif os_name =='windows':
        temp = image_file.split('/')
        image_id = int(temp[len(temp) - 1].split('\\')[1][:6])
    else:
        pass
    return image_id

def get_imagid_from_Blockimage(image_file):
    image_file_name = os.path.basename(image_file)
    temp = image_file_name.split('_')
    image_id = int(temp[0])
    height = int(temp[1])
    if len(temp)==3:
        width = int(temp[2].split('.')[0])
        block_number = 0
    else:
        width = int(temp[2])
        block_number = int(temp[3].split('.')[0])
    return image_id, height, width, block_number

def get_blockimage_area(height,width,block_number):
    if height == 2160 and width == 3840:
        # image are blocked to 36 image
        col = block_number // 6 # 列
        row = block_number % 6 # 行
        xmin,xmax = row*640,(row+1)*640
        ymin,ymax = col*360,(col+1)*360
    elif height == 1080 and width == 1920:
        col = block_number // 3
        row = block_number % 3
        xmin, xmax = row * 640, (row + 1) * 640
        ymin, ymax = col * 360, (col + 1) * 360
    else:
        xmin,xmax = 0,width
        ymin,ymax = 0,height
    return xmin,ymin,xmax,ymax

def get_label_in_blockimage(gt_bboxs,area_xmin,area_ymin,area_xmax,area_ymax):
    label = np.zeros((360,640)).astype(np.int32)
    for gt_bbox in gt_bboxs:
        if gt_bbox[3] > area_xmin and gt_bbox[4] > area_ymin and gt_bbox[1] < area_xmax and gt_bbox[2] < area_ymax:
            if gt_bbox[1] < area_xmin: xmin = 0
            else:xmin = gt_bbox[1]-area_xmin
            if gt_bbox[3] > area_xmax: xmax = 640
            else: xmax= gt_bbox[3]-area_xmin
            if gt_bbox[2] < area_ymin: ymin = 0
            else:ymin = gt_bbox[2]-area_ymin
            if gt_bbox[4] > area_ymax: ymax = 360
            else:ymax = gt_bbox[4]-area_ymin
            label[ymin:ymax,xmin:xmax] = gt_bbox[0]
    return label
