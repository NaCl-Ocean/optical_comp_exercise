from __init__ import *
import skimage.measure
import os
class_dict_reverse = {1:'holothurian', 2:'echinus', 3:'scallop', 4:'starfish'}

def mask_to_bbox(mask,cutoff_level,csv_writer,image_id):
    # get the bbox of image(image_id)  in class_i
    # write the result to csv_file
    prediction = np.zeroslike(mask).astype(int)
    prediction[mask <= cutoff_level] = 1
    objects = skimage.measure.label(prediction)
    for object in objects:
        confidence = get_confidence(prediction,object.bbox)
        csv_writer.writerow([class_dict_reverse[cutoff_level], image_id, confidence, object.bbox[0], object.bbox[1], object.bbox[2],object.bbox[3]])




def get_confidence(class_i_image,bbox):
    confidences = []
    mask = class_i_image[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    # 取像素中置信度最高为最终的置信度
    confidence = np.max(np.max(mask))
    return confidence


def Test_validate(test_data_file,dataset_mode,weights_path):
    '''
    :param dataset_mode: dataset from trian_dataset or test_dataset
    :param weights_path: the path to model weights
    :return:
    '''


    dataloader = dataset.get_Test_dataloaders(test_data_file,batch_szie=5)
    model = unet_model.UNet(3,5)
    model.load_state_dict(torch.load(weights_path))
    model  = model.cuda()
    model.eval()
    weights_dir = os.path.dirname(weights_path)
    os.makedirs(weights_dir + '/%s_prediction/' , exist_ok=True)
    with open(weights_dir+'/%s_prediction','w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_head = ['name','image_id','confidence','xmin','ymin','xmax','ymax']
        csv_writer.writerow(csv_head)
        for i,(images,image_ids) in enumerate(dataloader):
            outputs = model(images.cuda())
            outputs = outputs.detach().cpu()
            predictions = torch.argmax(outputs, dim=1)
            for j,prediction in enumerate(predictions):
               for i in range(5):
                   mask_to_bbox(prediction,i,csv_writer,image_ids[j])