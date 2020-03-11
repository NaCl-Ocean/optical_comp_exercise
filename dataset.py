from __init__ import *
import glob
from PIL import Image
import os
import bs4 as bs
import torchvision.transforms as transforms




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

        if augment:
            transform_list = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                              transforms.Resize((720, 425)), transforms.ToTensor()]
        else:
            transform_list=[transforms.Resize((720,425)),transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.len


    def get_image(self,i):
        i = i % len(self.files)
        image = np.array(Image.open(self.files[i]))
        height, width, channels = image.shape
        # channel last to channel first
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, height, width



    def __getitem__(self, i):
        image, height, width = self.get_image(i)


        label = get_label(self.gt_files[i],  height, width)

        # # 数据增强:水平翻转 or 竖直翻转
        # if self.augment: # random horizontal/vertical flips
        #     if np.random.random() < 0.5:
        #         image, label = np.flip(image,axis=-1), np.flip(label,axis=-1)
        #     if np.random.random() < 0.5:
        #         image, label = np.flip(image, axis=-2), np.flip(label, axis=-2)

        # channel last 修改为 channel last
        # np.moveaxis(image, 2, 0)
        # image = torch.from_numpy(image.copy())
        # torch.int64
        image = image.float()
        label = torch.LongTensor(label.copy())
        return image, label, i


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
        self.start_index = np.around(len(self.files)*train_size) + 1
        self.transform = transforms.Compose([transforms.Resize((720,425)),transforms.ToTensor()])

    def __len__(self):
        return self.len

    def get_image(self, i):
        i = int((i + self.start_index) % len(self.files))
        image = np.array(Image.open(self.files[i]))
        # channel last to channel first
        height, width, channels = image.shape
        # resize image
        image = Image.fromarray(image)
        image = self.transform(image)
        # channel last to channel first
        image = torch.squeeze(image,0)
        return image, height, width

    def __getitem__(self, i):
        i = int((i + self.start_index) % len(self.files))
        image, height, width = self.get_image(i)

        label = get_label(self.gt_files[i], height, width)

        image = image.float()
        # torch.int64
        label = torch.LongTensor(label.copy())
        return image, label, i


class TestDataset(torch.utils.data.Dataset):

    def __init__(self,image_data_dir):
        '''
        :param image_data_dir: path to image_data_dir
        '''
        self.files = sorted(glob.glob(image_data_dir, '/*.jpg'))
        self.len = len(self.files)

    def __len__(self):
        return len(self)

    def get_image(self, i):
        i = i % len(self.files)
        image = np.array(Image.open(self.files[i]))
        height, width, channels = image.shape
        # channel last to channel first
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, height, width

    def __getitem__(self, i):
        image, height, width = self.get_image(i)
        image = image.float()
        return image, i



def get_gt_box(gt_box_file,height,width):
    with open(gt_box_file, 'r') as f:
        boxContent = f.read()
    soup = bs.BeautifulSoup(boxContent, features='xml')
    objects = soup.findAll("object")
    gt_boundingbox = np.zeros((len(objects), 5)).astype(np.int32)
    for i, object in enumerate(objects):
        if object.contents[1].string != 'waterweeds':
            gt_boundingbox[i, 0] = class_dict[object.contents[1].string]
            gt_boundingbox[i, 1] = int(object.bndbox.xmin.getText())/ width * 425
            gt_boundingbox[i, 2] = int(object.bndbox.ymin.getText())/ height * 720
            gt_boundingbox[i, 3] = int(object.bndbox.xmax.getText())/ width * 425
            gt_boundingbox[i, 4] = int(object.bndbox.ymax.getText())/ height * 720
    return gt_boundingbox


def get_label(gt_box_file, height, width):
    # label (H,W)
    label = np.zeros((720, 425)).astype((np.int32))
    gt_bbox = get_gt_box(gt_box_file, height, width)

    for i in range(gt_bbox.shape[0]):
        label[gt_bbox[i, 2]:gt_bbox[i, 4], gt_bbox[i, 1]:gt_bbox[i, 3]] = gt_bbox[i, 0]
    return label



def get_train_datasets(train_data_file,train_gt_file,augment=False, train_size=0.8):
    return TrainDataset(train_data_file, train_gt_file, augment=augment,train_size=train_size)

def get_train_dataloaders(train_data_file,train_gt_file,batch_size,augment=False,shuffle=False,train_size=0.8):
    train_dataset = get_train_datasets(train_data_file=train_data_file,train_gt_file=train_gt_file,augment=augment,train_size=train_size)
    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloder

def get_val_datasets(train_data_file,train_gt_file,train_size=0.8):
    return ValDataset(train_data_file,train_gt_file,train_size=train_size)

def get_Val_dataloaders(train_data_file,train_gt_file,batch_size,shuffle=False,train_size=0.8):
    val_dataset  = get_val_datasets(train_data_file=train_data_file,train_gt_file=train_gt_file,train_size=train_size)
    val_dataloaders = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=shuffle)
    return val_dataloaders

def get_test_datasets(test_data_file):
    return TestDataset(test_data_file)

def get_Test_dataloaders(test_data_file,batch_size):
    test_dataset = get_test_datasets(test_data_file)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return test_dataloader