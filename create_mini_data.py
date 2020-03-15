import os
import glob
import random
import shutil
save_dir = 'data/'
save_image_dir = save_dir+'image'
save_box_dir = save_dir+'box'

print(random.randint(1,56))
image_data_dir = '../watert_comp_exercise/data/train/train/image'
box_data_dir = '../watert_comp_exercise/data/train/train/box'

image_files = glob.glob(image_data_dir+'/*.jpg')
box_files = glob.glob(box_data_dir+'/*.xml')
for i in range(len(image_files)):
    randomnumber = random.randint(1,56)
    if  randomnumber == 1 :
        shutil.copy(image_files[i], save_image_dir)
        shutil.copy(box_files[i], save_box_dir)
