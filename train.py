from __init__ import *
import unet_model
import dataset
import torch.nn as nn
from skimage import measure
import csv

# 每一类对于loss的weight
loss_weight = np.array([0.25,0.25,0.25,0.25])

class_dict ={'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}
class_dict_reverse = {1:'holothurian', 2:'echinus', 3:'scallop', 4:'starfish'}

# parameters
args = {'train_ratio': 0.8, 'shuffle': False, 'augment': True,
        'momentum_factor': 0.9, 'learning_rate': 1e-4, 'weight_decay':1e-4, 'grad_clip_by_value':1.0,
        'batch_size': 3, 'epochs': 60,
        'in_channels': 3, 'n_classes': 5,
        'save':True, 'save_freq':20,
        'confidence_threshold': 0.5}

class Trainer():
    def __init__(self, dir_dict):
        self.batch_size = int(args['batch_size'])
        self.augment = args['augment']
        self.train_ratio = float(args['train_ratio'])
        self.shuffle = args['shuffle']
        self.n_classes = int(args['n_classes'])
        self.n_channels = int(args['in_channels'])
        self.learning_rate = float(args['learning_rate'])
        self.epochs = int(args['epochs'])
        self.momentum_factor = float(args['momentum_factor'])
        self.weight_decay = float(args['weight_decay'])
        self.grid_clip_by_value = float(args['grad_clip_by_value'])
        self.save = args['save']
        self.save_freq = int(args['save_freq'])
        self.save_dir = dir_dict['save_dir']
        self.confidence_threshold = float(dir_dict['confidence_threshold'])
        self.csv_dir = dir_dict['csv_dir']
        self.bulid_dataset()
        self.build_model()

    def build_model(self):
        self.model = unet_model.UNet(self.n_channels, self.n_classes)

    def bulid_dataset(self):
        self.train_dataloader = dataset.get_train_dataloaders(batch_size= self.batch_size, augment=self.augment, shuffle=self.shuffle, train_size=self.train_ratio)
        self.val_dataloader = dataset.get_Val_dataloaders(batch_size= self.batch_size, shuffle=self.shuffle, train_size=self.train_ratio)

    def loss(self, output, label):
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight))

        # output:(B,n,H,W) to (B,H,W,n)
        # label:(B,H,W) [0,3]
        # label 对应 output的通道 0->通道0 1->通道1 2->通道2 3->通道3
        # output 的通道0表示label=0的概率分布
        for i, j in [[1,2],[2,3]]:
            # 转置操作 类似于numpy.moveaxis
            output = torch.transpose(output, i, j)
        loss = loss_function(output.contiguous(), label)
        return loss


    def build_optimiezer(self):
        model = self.model
        parameters = model.parameters()
        self.optimizer = torch.optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum_factor, weight_decay=self.weight_decay)

    def train_epoch(self):
        model, train_loader, optimizer = self.model, self.train_dataloader, self.optimizer
        model.train()
        steps = 0
        for i, (image,label) in  enumerate(train_loader):
            image = image
            label = label
            output = model(image)
            loss = self.loss(output= output, label= label)
            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-self.grid_clip_by_value, self.grid_clip_by_value)

            optimizer.step()
            for b in range(self.batch_size):
                steps = steps + 1
                output_b = output[b,:,:,:]
                for j in range(1,self.n_classes+1):
                    class_j_image = output_b[j,:,:]






    def save_checkpoints(self, epoch):
        model = self.model
        if self.save and epoch % self.save_freq == 0 :
            torch.save(self.model.state_dict(), self.save_dir, '/epoch%s.pth' % str(epoch).zfill(3))
        elif epoch == self.epochs:
            torch.save(model.state_dict(), self.save_dir + '/epoch%s.pth' % str(epoch).zfill(3))

    def train(self):
        self.model = self.model


        self.model.trian()
        for i in range(1, self.epochs+1):
            self.train_epoch()
            self.save_checkpoints(epoch = i)




    def get_bbox(self, class_i_image,class_i,steps):
        bboxs = []

        class_i_image = np.numpy(class_i_image)
        class_i_image = np.where(class_i_image >= self.confidence_threshold, 1, 0)
        lbl = measure.label(class_i_image)
        props = measure.regionprops(lbl)
        # bbox x_min,y_min,x_max,y_max
        for prop in props:
            bboxs.append([class_dict_reverse[class_i], str("%06d" % steps)])
        return bboxs

    def write_csv(self, image_id, class_i, csv_file_path):
        csv_file_path = self.csv_dir + ''





    def map_metric(self):

    def validate_epoch(self):


