import os
import random
import cv2
import torch
import math
import time
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utilFuctions import showMultipleImgs


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


#  load the trained models for prediction
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PATH = r'E:\大四下\毕设\第四周_MaskRCNN\trained_model\model_cyto_nuclei_403.pt'
# change the number accordingly
model = get_instance_segmentation_model(num_classes=3)
model.load_state_dict(torch.load(PATH))
model.to(device)

# put the model in evaluation mode
model.eval()

# class map for 4-class segmentation assignments
# class_name_to_value = {'_background_':0, 'N': 1, 'L': 2, 'H': 3}
# class_vlaue_to_name = {0:'_background', 1:'N', 2:'L', 3:'H'}

# class map for 3-class segmentation and 2-class segmentation
class_name_to_value = {'_background_': 0, 'Cytoplasm': 1, 'Nuclei': 2}
class_vlaue_to_name = {0: '_background', 1: 'Cytoplasm', 2: 'Nuclei'}


# extract prediction information from the model output
def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    with torch.no_grad():
        pred = model([img.to(device)])
    print('pred')
    print(pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(pred_t) == 0:
        pred_t = [pred_score.index(np.max(pred_score))][-1]
    else:
        pred_t = pred_t[-1]
    print("masks>0.5")
    print(pred[0]['masks'] > 0.5)
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    print("this is masks")
    print(masks)
    # pred_class = [class_vlaue_to_name[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
    # rewrite the pred_class and pred_labels variables
    pred_class = list(pred[0]['labels'].detach().cpu().numpy())
    pred_labels = [class_vlaue_to_name[i] for i in pred_class]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class, pred_labels


# to fill the mask with different colors
def random_colour_masks(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = random_color()
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


# get a random color from the color list of 11 colors, and more can be added if necessary
def random_color():
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    return colours[random.randrange(0, 10)]


def img_seg_contour(img_path, threshold=0.5, con_th=3, rect_th=2, text_size=1, text_th=2):
    masks, boxes, _, pred_labels = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        # the following lines are used to fill the masks with color, instead of delineate the contours of them
        # rgb_mask = random_colour_masks(masks_contours)
        # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        # img = cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        # cv2.putText(img, pred_labels[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0),thickness=text_th)

        # these for delineating contours
        masks_contours = measure.find_contours(masks[i], 0.5)
        for contour in masks_contours:
            contour = np.asarray(contour).astype(np.int32)
            contour[:, [0,1]] = contour[:, [1,0]]
            contour = contour.reshape((-1, 1, 2))
            img = cv2.polylines(img, [contour], True, color=random_color(), thickness=con_th)
    return img


# this function is to check if the pred cytos have nucleis inside or nucleis in cytos
def check_cyto(cyto_box, nuclei_boxes):
    nuclei_num = 0
    for nuclei_box in nuclei_boxes:
        if cyto_box[0][0] <= nuclei_box[0][0] and cyto_box[0][1] <= nuclei_box[0][1] and cyto_box[1][0] \
                >= nuclei_box[1][0] and cyto_box[1][1] >= nuclei_box[1][1]:
            nuclei_num += 1
    return nuclei_num


# this function is to check if the pred nucleis have cytos outside
def check_nuclei(nuclei_box, cyto_boxes):
    cyto_num = 0
    for cyto_box in cyto_boxes:
        if cyto_box[0][0] <= nuclei_box[0][0] and cyto_box[0][1] <= nuclei_box[0][1] and cyto_box[1][0] \
                >= nuclei_box[1][0] and cyto_box[1][1] >= nuclei_box[1][1]:
            cyto_num += 1
    return cyto_num


# this function will return all the predicted instances of the input image separately
# including the cytoplasm and nuclei of the cells
def get_instance(img_path):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    masks, pred_boxes, pred_class, pred_labels = get_prediction(img_path, threshold=0.5)
    cyto_index = [i for i, x in enumerate(pred_class) if x == 1]
    nuclei_index = [i for i, x in enumerate(pred_class) if x == 2]
    cyto_boxes = [pred_boxes[i] for i in cyto_index]
    nuclei_boxes = [pred_boxes[i] for i in nuclei_index]
    masks = torch.from_numpy(masks)
    print(img.size())
    cytos = []
    nucleis = []
    print(masks.size())
    for i in range(len(masks)):
        if pred_class[i] == 1:
            cytos.append(img.mul(masks[i]).squeeze().detach().cpu().numpy())
        elif pred_class[i] == 2:
            nucleis.append(img.mul(masks[i]).squeeze().detach().cpu().numpy())
    for i in range(len(cyto_boxes)):
        if not check_cyto(cyto_boxes[i], nuclei_boxes):
            del cytos[i]
    for i in range(len(nuclei_boxes)):
        if not check_nuclei(nuclei_boxes[i], cyto_boxes):
            del nucleis[i]
    return cytos, nucleis


imgs_path = os.listdir(r'dataset\isbi15_edf')
imgs = []
for img_path in imgs_path:
    img_path = os.path.join(r'dataset\isbi15_edf', img_path)
    imgs.append(img_seg_contour(img_path))

cytos_test, nucleis_test = get_instance(os.path.join(r'dataset\isbi15_edf', imgs_path[0]))
for i in range(len(cytos_test)):
    showMultipleImgs(cytos_test[i], 1, 1)
