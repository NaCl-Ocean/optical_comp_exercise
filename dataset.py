from __init__ import *
import glob
from PIL import Image
import os
import bs4 as bs




main_dir = os.getcwd()

TRAIN_FILES_PATH = 'E:/02竞赛/水下目标检测/water_optical_comp/train/train/image'
TRAIN_GT_XML_PATH = 'E:/02竞赛/水下目标检测/water_optical_comp/train/train/box'

#TRAIN_FILES_PATH = main_dir + '/water_optical_comp/train/train/image'
#TRAIN_GT_XML_PATH = main_dir + '/water_optical_comp/train/train/box'

class_dict ={'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}
# class应该有5类，四类为目标，另一类为背景

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, gt_file_path, augment=False, train_size=0.8):
        '''
           file_path:image_file_path
           gt_file_path : path to ground truth file(xml)
           train_size: Proportion of training set in all data
        '''
        self.files = sorted(glob.glob(file_path + '/*.jpg'))
        self.gt_files = sorted(glob.glob(gt_file_path + '/*.xml'))
        self.len =  np.around(len(self.files)*train_size).astype(np.int32)
        self.augment = augment

    def __len__(self):
        return self.len


    def get_image(self,i):
        i = i % len(self.files)
        image = np.array(Image.open(self.files[i]))
        height, width, channels = image.shape
        # channel last to channel first
        image = np.moveaxis(image,2,0)
        image = image.reshape((channels, height, width)).astype(np.float32)
        return image, height, width


    def __getitem__(self, i):
        image, height, width = self.get_image(i)


        label = get_label(self.gt_files[i],  height, width)

        # 数据增强:水平翻转 or 竖直翻转
        if self.augment: # random horizontal/vertical flips
            if np.random.random() < 0.5:
                image, label = np.flip(image,axis=-1), np.flip(label,axis=-1)
            if np.random.random() < 0.5:
                image, label = np.flip(image, axis=-2), np.flip(label, axis=-2)

        # channel last 修改为 channel last
        np.moveaxis(image, 2, 0)
        image = torch.from_numpy(image.copy())
        # torch.int64
        label = torch.LongTensor(label.copy())
        return image, label


class ValDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, gt_file_path, train_size=0.8):
        '''
           file_path:image_file_path
           gt_file_path : path to ground truth file(xml)
           train_size: Proportion of training set in all data
        '''
        self.files = sorted(glob.glob(file_path + '/*.jpg'))
        self.gt_files = sorted(glob.glob(gt_file_path + '/*.xml'))
        self.len = np.around(len(self.files)*(1-train_size)).astype((np.int32))
        self.start_index = len(self.files) - np.around(len(self.files)*train_size) + 1

    def __len__(self):
        return self.len

    def get_image(self, i):
        i = (i + self.start_index) % len(self.files)
        image = np.array(Image.open(self.files[i]))
        # channel last to channel first
        height, width, channels = image.shape
        image = image.reshape((channels, height, width)).astype(np.float32)
        return image, height, width

    def __getitem__(self, i):
        i = (i + self.start_index) % len(self.files)
        image, height, width = self.get_image(i)

        label = get_label(self.gt_files[i], height, width)

        image = torch.from_numpy(image.copy())
        # torch.int64
        label = torch.LongTensor(label.copy())
        return image, label





def get_gt_box(gt_box_file):
    with open(gt_box_file, 'r') as f:
        boxContent = f.read()
    soup = bs.BeautifulSoup(boxContent, features='xml')
    objects = soup.findAll("object")
    gt_boundingbox = np.zeros((len(objects), 5)).astype(np.int32)
    for i, object in enumerate(objects):
        gt_boundingbox[i, 0] = class_dict[object.contents[1].string]
        gt_boundingbox[i, 1] = int(object.bndbox.xmin.getText())
        gt_boundingbox[i, 2] = int(object.bndbox.ymin.getText())
        gt_boundingbox[i, 3] = int(object.bndbox.xmax.getText())
        gt_boundingbox[i, 4] = int(object.bndbox.ymax.getText())
    return gt_boundingbox


def get_label(gt_box_file, height, width):
    label = np.zeros((1,height, width)).astype((np.int32))
    gt_bbox = get_gt_box(gt_box_file)
    for i in range(gt_bbox.shape[0]):
        label[0, gt_bbox[i, 2]:gt_bbox[i, 4], gt_bbox[i, 1]:gt_bbox[i, 3]] = gt_bbox[i, 0]
    return label



def get_train_datasets(augment=False, train_size=0.8):
    return TrainDataset(TRAIN_FILES_PATH, TRAIN_GT_XML_PATH, augment=augment,train_size=train_size)



def get_train_dataloaders(batch_size,augment=False,shuffle=False,train_size=0.8):
    train_dataset = get_train_datasets(augment=augment,train_size=train_size)
    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloder

def get_val_datasets(train_size=0.8):
    return ValDataset(TRAIN_FILES_PATH,TRAIN_GT_XML_PATH,train_size=train_size)

def get_Val_dataloaders(batch_size,shuffle=False,train_size=0.8):
    val_dataset  = get_val_datasets(train_size=train_size)
    val_dataloaders = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=shuffle)
    return val_dataloaders











