from __init__ import *
import unet_model
import dataset
import torch.nn as nn
from skimage import measure
import datetime
import csv

# 每一类对于loss的weight
loss_weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])




class Trainer():
    def __init__(self, dir_dict, args):
        self.args=args
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
        self.confidence_threshold = float(args['confidence_threshold'])
        self.lr_decay_step = int(args['lr_decay_step'])
        self.dir_dict = dir_dict
        # to GPU
        self.cuda_gpu = torch.cuda.is_available()
        # load dataset
        self.bulid_dataset()
        # build model
        if self.cuda_gpu:
            self.model = unet_model.UNet(self.n_channels, self.n_classes).cuda()
        else:
            self.model = unet_model.UNet(self.n_channels, self.n_classes)
        # build optimizer
        self.build_optimiezer()




    def bulid_dataset(self):
        self.train_dataloader = dataset.get_train_dataloaders(self.dir_dict['train_image_dir'],self.dir_dict['train_box_dir'],batch_size= int(self.args['batch_size']), augment=self.augment, shuffle=self.shuffle, train_size=self.train_ratio)
        self.val_dataloader = dataset.get_Val_dataloaders(self.dir_dict['train_image_dir'],self.dir_dict['train_box_dir'],batch_size= int(self.args['batch_size']), shuffle=self.shuffle, train_size=self.train_ratio)

    def loss(self, output, label):
        if self.cuda_gpu:
            loss_function = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight).float().cuda())
        else:
            loss_function = nn.CrossEntropyLoss(weight=torch.tensor(loss_weight).float())

        # output:(B,n,H,W) to (B,H,W,n)
        # label:(B,H,W) [0,3]
        # label 对应 output的通道 0->通道0 1->通道1 2->通道2 3->通道3
        # output 的通道0表示label=0的概率分布
        loss = loss_function(output, label)
        return loss


    def build_optimiezer(self):
        model = self.model
        parameters = model.parameters()
        self.optimizer = torch.optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum_factor, weight_decay=self.weight_decay)

    def train(self):
        self.model = self.model
        self.model.train()
        Map_result_file = self.dir_dict['result_dir'] + '/'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'
        with open(Map_result_file,'w') as Map_result_file:
            Map_writer = csv.writer(Map_result_file)
            # Map_header = ['epoch','train_Map','train_loss','val_Map','val_loss']
            Map_header = ['epoch','train_loss','val_loss']
            Map_writer.writerow(Map_header)

            for i in range(1, self.epochs + 1):\

                print('train epoch',i)
                train_csv_path, train_loss = self.train_epoch(i)
                print('epoch'+str(i)+'train'+str(train_loss))

                self.save_checkpoints(epoch=i)
                #train_Map = Map_metric.Map_eval(self.dir_dict['train_box_dir'],train_csv_path,self.n_classes)
                print('validate epoch',i)
                val_csv_path, val_loss=self.validate_epoch(i)
                print('epoch' + str(i) + 'validata' + str(val_loss))
                #val_Map = Map_metric.Map_eval(self.dir_dict['train_box_dir'],val_csv_path,self.n_classes)
                # write the train map and validate map to csv file
                Map_writer.writerow([i,train_loss,val_loss])


    def train_epoch(self,epoch):
        model, train_loader, optimizer = self.model, self.train_dataloader, self.optimizer
        model.train()
        losses = AverageMeter()
        self.adjust_learning_rate(epoch)
        csv_file_path = self.dir_dict['csv_dir'] + '/'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_train_' + str("%06d" % epoch ) + '.csv'
        with open(csv_file_path,'w',newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_head = ['name','image_id','confidence','xmin','ymin','xmax','ymax']
            csv_writer.writerow(csv_head)

            for i, (image,label,image_ids) in  enumerate(train_loader):
            # batch
                if self.cuda_gpu:
                    image = image.cuda()
                    label = label.cuda()
                    output = model(image)
                else:
                    image = image
                    label = label
                    output = model(image)
                loss = self.loss(output=output, label= label)
                output = output.cpu()
                losses.update(float(loss.data.item()), image.size(0))
                optimizer.zero_grad()
                loss.backward()

                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-self.grid_clip_by_value, self.grid_clip_by_value)

                optimizer.step()
                if i % 100 == 0:
                    print('finish training image',image_ids)
                self.write_det_csv(image_ids, csv_writer=csv_writer, output=output)

            # close the csv file to read the csv file
            csvfile.close()
        return csv_file_path, losses.avg




    def save_checkpoints(self, epoch):
        model = self.model
        if self.save and epoch % self.save_freq == 0 :
            torch.save(self.model.state_dict(), self.dir_dict['save_dir'] + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + str("%06d" % epoch) +'.pth')
        elif epoch == self.epochs:
            torch.save(model.state_dict(), self.dir_dict['save_dir'] + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + str("%06d" % epoch) +'.pth')




    def get_confidence(self,class_i_image,bbox):
        confidences = []
        mask = class_i_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # 取像素中置信度最高为最终的置信度
        confidence = np.max(np.max(mask))
        return confidence

    def validate_epoch(self, epoch):
        val_loader, model = self.val_dataloader, self.model
        model.eval()
        losses = AverageMeter()
        # detection result
        csv_file_path = self.dir_dict['csv_dir'] +'/'+ datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + \
                        '_val_' + str("%06d" % epoch) + '.csv'
        with open(csv_file_path,'w') as CSVfile:
            csv_writer = csv.writer(CSVfile)
            csv_head = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
            csv_writer.writerow(csv_head)
            for i,(image,label,image_ids) in  enumerate(val_loader):
                if self.cuda_gpu:
                    image = image.cuda()
                    label = label.cuda()
                    output = model(image)
                else:
                    image = image
                    label = label
                    output = model(image)
                loss = self.loss(output,label)
                output = output.cpu()
                losses.update(float(loss.data.item()),image.size(0))
                if i % 100 == 0:
                    print('finish validating image',image_ids)
                self.write_det_csv(image_ids, csv_writer=csv_writer, output= output)
            CSVfile.close()

        return csv_file_path, losses.avg






    def write_det_csv(self,image_ids,csv_writer,output):
        for b in range(list(image_ids.size())[0]):
            # 处理单个batch中sample data
            image_id = image_ids[b].numpy()
            output_b = output[b, :, :, :]
            for j in range(1, self.n_classes):
                # 处理sample data中的单个通道（class)
                class_j_image = output_b[j, :, :]
                csv_content = self.get_bbox(class_j_image, j, image_id,csv_writer=csv_writer)


    def get_bbox(self, class_i_image, class_i, image_id,csv_writer):

        csv_content = []

        class_i_image_numpy = class_i_image.detach().numpy()
        binary_image = np.where(class_i_image_numpy >= self.confidence_threshold, 1, 0)
        # 连通域分析
        # mask to bbox
        # 效果不太好
        lbl = measure.label(binary_image)
        props = measure.regionprops(lbl)
        # bbox x_min,y_min,x_max,y_max
        for prop in props:
            bbox = prop.bbox
            confidence = self.get_confidence(class_i_image_numpy, bbox)
            csv_writer.writerow([class_dict_reverse[class_i], str("%06d" % (image_id+1))+'.xml', confidence, bbox[0], bbox[1], bbox[2], bbox[3]])



    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 after args.lr_decay_step steps"""
        lr = self.learning_rate * (0.1 ** (epoch // self.lr_decay_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count







