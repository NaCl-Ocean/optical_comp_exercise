import os


def get_dir():
    main_dir = os.path.dirname(__file__)
    save_dir = main_dir + '/weights'
    data_dir = main_dir + '/data'
    train_dir = data_dir + '/train/train'
    train_image_dir = train_dir + '/image'
    train_box_dir = train_dir + '/box'
    csv_dir = main_dir + '/prediction'
    return {'save_dir': save_dir, 'train_image_dir': train_image_dir, 'train_box_dir': train_box_dir,'csv_dir':csv_dir}
if __name__ == '__main__':
    dir_dict = get_dir()

