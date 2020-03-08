from __init__ import *
import csv
import glob
import dataset


class_name_dict=['holothurian', 'echinus', 'scallop', 'starfish']
def get_gt(train_box_dir):
    recs = {}
    files = sorted(glob.glob(train_box_dir + '/*.xml'))
    for i in len(files):
        gt_box_file = files[i]
        gt_box = dataset.get_gt_box(gt_box_file)
        class_list = [class_name_dict[gt_box[j,0]-1] for j in range(np.size(gt_box,0)) ]
        recs[i] = {'bbox':gt_box[:,1:4],
                         'class':class_list}
    return recs

def get_class_i_in_image_j(recs,class_i,image_j):
    objects_image_j = recs[image_j]
    class_i_bboxs = []
    object_num = len(objects_image_j['class'])
    detected = [False] * object_num
    for i in range(object_num):
        if objects_image_j['class'][i] == class_i:
            class_i_bboxs.append(objects_image_j['bbox'][i])
    return np.array(class_i_bboxs), detected

def get_class_i_objects(recs,class_i):
    num = 0
    for i in range(len(recs)):
        num = num + len([j for j in range(len(recs[i]['class'])) if recs[i]['class'][j]==class_i ])
    return num


def get_det(csv_file):
    with open(csv_file,'r') as csvFile:
        csvreader = csv.reader(csvFile)

    imageids_list = []
    bboxs_list = []
    class_list=[]
    confidence_list=[]
    for item in csvreader:
        if csvreader.line_num == 1:
            continue
        imageids_list.append(item[1])
        bboxs_list.append(item[3:])
        class_list.append(item[0])
        confidence_list.append(item[2])
    return imageids_list,bboxs_list,class_list,confidence_list

def get_det_class(class_i,imageids_list,bboxs_list,class_list,confidence_list):
    class_i_index = [i for i in range(len(class_list)) if class_list[i] == class_i]
    bboxs = np.array(bboxs_list)
    confidences = np.array(confidence_list)
    imageids = np.array(imageids_list)
    class_i_bboxs = bboxs[class_i_index]
    class_i_confidence = confidences[class_i_index]
    class_i_imageids = imageids[class_i_index]
    return class_i_imageids,class_i_bboxs,class_i_confidence

def voc_eval(train_box_dir,csv_file_path,n_classes,IOU_THRESHOLD=0.5):
    recs = get_gt(train_box_dir=train_box_dir)
    imageids_list, bboxs_list, class_list, confidence_list = get_det(csv_file_path)
    map = 0
    # 开始处理每个class
    for i in range(n_classes):
        class_i_imageids,class_i_bboxs,class_i_confidence= get_det_class(class_name_dict[i], imageids_list, bboxs_list, class_list, confidence_list)
        sorted_index = np.argsort(-class_i_confidence)
        class_i_bboxs = class_i_bboxs[sorted_index]
        class_i_imageids = class_i_imageids[sorted_index]


        nd = len(class_i_imageids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        # 开始处理每个object
        for d in range(nd):
            # 获取该object在哪张图像
            image_id = class_i_imageids[d]
            DETbbox = class_i_bboxs[d]
            confidence = class_i_confidence[d]
            GTbboxs,detected = get_class_i_in_image_j(recs,class_name_dict[i],image_id)
            IOU,overlapmaxid = IOU_eval(GTbboxs,DETbbox)
            if IOU > IOU_THRESHOLD:
                # if IOU>threshold
                if not detected[overlapmaxid]:
                # 如果该true_box没有被探测到
                    tp[d]=1
                    detected[overlapmaxid]=True
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/get_class_i_objects(recs,class_name_dict[i])
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, False)
        map= map+ap
    return map/n_classes



def IOU_eval(BBgt, BBdet):
    # BoundingBox.format:[xmin,ymin.xmax,ymax]
    #
    if (isinstance(BBgt, (np.ndarray)) and isinstance(BBdet, (np.ndarray))):
        xmin = np.maximum(BBgt[:, 0], BBdet[0]);
        ymin = np.maximum(BBgt[:, 1], BBdet[1]);
        xmax = np.minimum(BBgt[:, 2], BBdet[2]);
        ymax = np.minimum(BBgt[:, 3], BBdet[3]);

        width = np.maximum(xmax - xmin + 1, 0.)
        height = np.maximum(ymax - ymin + 1, 0.)

        # evalaute area of inters
        inters = width * height

        # evalute area of union
        union = ((BBdet[2] - BBdet[0] + 1.) * (BBdet[3] - BBdet[1] + 1.) +
                 (BBgt[:, 2] - BBgt[:, 0] + 1.) * (BBdet[:, 3] - BBgt[:, 1] + 1.) -
                 inters)

        # evalute IOU
        overlaps = inters / union
        overlapmax = np.maximum(overlaps)
        overlapmax_index = np.argmax(overlaps)

        return overlapmax, overlapmax_index



def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

