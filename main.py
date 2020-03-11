import os
from __init__ import *


# parameters
args = {'train_ratio': 0.8, 'shuffle': False, 'augment': True,
        'momentum_factor': 0.9, 'learning_rate': 1e-4, 'weight_decay':1e-4,
        'grad_clip_by_value':1.0, 'lr_decay_step': 40,
        'batch_size': 3, 'epochs': 60,
        'in_channels': 3, 'n_classes': 5,
        'save':True, 'save_freq':20,
        'confidence_threshold': 0.5}

def get_dir():
    """
    save_dir: save model weights
    data_dir: data for training and validating
    csv_dir: save detection result(format csv)
    result_dir: save loss every epoch(train and val)
    """
    main_dir = os.path.dirname(__file__)
    save_dir = main_dir + '/weights'
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    data_dir = main_dir + '/data'
    train_dir = data_dir + '/train/train'
    train_image_dir = train_dir + '/image'
    train_box_dir = train_dir + '/box'
    csv_dir = main_dir + '/detection'
    if not os.path.exists(csv_dir): os.mkdir(csv_dir)
    result_dir = main_dir +'/result'
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    return {'save_dir': save_dir, 'train_image_dir': train_image_dir, 'train_box_dir': train_box_dir,
            'csv_dir':csv_dir, 'result_dir':result_dir}



if __name__ == '__main__':
    dir_dict = get_dir()
    trainer = train.Trainer(dir_dict,args=args)
    trainer.train()

