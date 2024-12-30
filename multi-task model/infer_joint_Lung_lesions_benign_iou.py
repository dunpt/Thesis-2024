# loading in and transforming data
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from skimage import io, transform
from PIL import Image

# visualizing data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import cv2
# load dataset information
import yaml

import json
# image writing
import imageio
from skimage import img_as_ubyte

# Clear GPU cache
torch.cuda.empty_cache()

import argparse
from pathlib import Path

parser = argparse.ArgumentParser("ESFPNet based model")
# parser.add_argument('--label_json_path', type=str, required=True,
#         help='Location of the data directory containing json labels file of each task after combining two json files.json')
# parser.add_argument('--path_imgs_test', type=str, required=True,
#         help='Location of the images of test_phase for each tasks)')
# parser.add_argument('--path_masks_test', type=str, required=True,
#         help='Location of the masks of test_phase for each tasks)')
# parser.add_argument('--model_type', type=str, default='B4',
#         help='Type of model (default B4)')
# parser.add_argument('--init_trainsize', type=int, default=352,
#         help='Size of image for training (default = 352)')
# parser.add_argument('--saved_model', type=str, required=True,
#         help='Load saved model') 

parser.add_argument('--label_json_path', type=str, default='/home/dunpt1504/Documents/School_project/Uncertainty/labels/labels_TTKV.json',
        help='Location of the data directory containing json labels file of each task after combining two json files.json')
parser.add_argument('--path_imgs_test', type=str, default='/home/dunpt1504/Documents/School_project/Uncertainty/test_dataset_qs/imgs',
        help='Location of the images of test_phase for each tasks)')
parser.add_argument('--path_masks_test', type=str, default='/home/dunpt1504/Documents/School_project/Uncertainty/test_dataset_qs/masks',
        help='Location of the masks of test_phase for each tasks)')
parser.add_argument('--model_type', type=str, default='B4',
        help='Type of model (default B4)')
parser.add_argument('--init_trainsize', type=int, default=352,
        help='Size of image for training (default = 352)')
parser.add_argument('--saved_model', type=str, default='/home/dunpt1504/Documents/School_project/Uncertainty/SaveModel/Lung_cancer_lesions_no_object_500_800/Mean_best.pt',
        help='Load saved model') 


args = parser.parse_args()

class test_dataset:
    def __init__(self, image_root, gt_root,label_root,  testsize): #
        self.testsize = 480#testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        print(len(self.images))
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        print(len(self.gts))
        self.labels = label_root
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
                                 ])
        self.transform_qs = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                      [0.229, 0.224, 0.225])
                                 ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image_qs = self.transform_qs(image).unsqueeze(0)
        image = self.transform(image).unsqueeze(0)
        
        gt = self.binary_loader(self.gts[self.index])
        file_name = os.path.splitext(os.path.basename(self.images[self.index]))[0]

        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        label_list = ['Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina', 'Mucosal infiltration', 'Vascular growth', 'Tumor', 'Benign']

        label_name = [file['label_name'] for file in data if file['object_id'] == file_name]
        
        label_tensor = torch.zeros([8])
        for name in label_name:
            label_tensor[label_list.index(name)] = 1
        
        
        self.index += 1
        return image, label_tensor, image_qs, file_name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
from collections import OrderedDict
import copy

from Encoder import mit
from Decoder import mlp
from mmcv.cnn import ConvModule

class ESFPNetStructure(nn.Module):

    def __init__(self, embedding_dim = 160):
        super(ESFPNetStructure, self).__init__()

        # Backbone
        if args.model_type == 'B0':
            self.backbone = mit.mit_b0()
        if args.model_type == 'B1':
            self.backbone = mit.mit_b1()
        if args.model_type == 'B2':
            self.backbone = mit.mit_b2()
        if args.model_type == 'B3':
            self.backbone = mit.mit_b3()
        if args.model_type == 'B4':
            self.backbone = mit.mit_b4()
        if args.model_type == 'B5':
            self.backbone = mit.mit_b5()

        self._init_weights()  # load pretrain

        # LP Header
        self.LP_1 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_2 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_3 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])
        self.LP_4 = mlp.LP(input_dim = self.backbone.embed_dims[3], embed_dim = self.backbone.embed_dims[3])

        # Linear Fuse
        self.linear_fuse34 = ConvModule(in_channels=(self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), out_channels=self.backbone.embed_dims[2], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=(self.backbone.embed_dims[1] + self.backbone.embed_dims[2]), out_channels=self.backbone.embed_dims[1], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=(self.backbone.embed_dims[0] + self.backbone.embed_dims[1]), out_channels=self.backbone.embed_dims[0], kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        # Fused LP Header
        self.LP_12 = mlp.LP(input_dim = self.backbone.embed_dims[0], embed_dim = self.backbone.embed_dims[0])
        self.LP_23 = mlp.LP(input_dim = self.backbone.embed_dims[1], embed_dim = self.backbone.embed_dims[1])
        self.LP_34 = mlp.LP(input_dim = self.backbone.embed_dims[2], embed_dim = self.backbone.embed_dims[2])

        # Final Linear Prediction
        self.linear_pred = nn.Conv2d((self.backbone.embed_dims[0] + self.backbone.embed_dims[1] + self.backbone.embed_dims[2] + self.backbone.embed_dims[3]), 1, kernel_size=1)

        #classification layer
        self.norm1 = nn.BatchNorm2d(512, eps=1e-5)
        self.Relu = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout(p=0.3)
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2d(256, eps=1e-5)
        self.conv2 = nn.Conv2d(256, 8, 1, stride=1, padding=0, bias=True) # 9 = number of classes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        

    def _init_weights(self):

        if args.model_type == 'B0':
            pretrained_dict = torch.load('./Pretrained/mit_b0.pth')
        if args.model_type == 'B1':
            pretrained_dict = torch.load('./Pretrained/mit_b1.pth')
        if args.model_type == 'B2':
            pretrained_dict = torch.load('./Pretrained/mit_b2.pth')
        if args.model_type == 'B3':
            pretrained_dict = torch.load('./Pretrained/mit_b3.pth')
        if args.model_type == 'B4':
            pretrained_dict = torch.load('./Pretrained/mit_b4.pth')
        if args.model_type == 'B5':
            pretrained_dict = torch.load('./Pretrained/mit_b5.pth')

        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        print("successfully loaded!!!!")


    def forward(self, x):

        ##################  Go through backbone ###################

        B = x.shape[0]

        #stage 1
        # print(x)
        out_1, H, W = self.backbone.patch_embed1(x)
        # print("out1", out_1)
        # print("self.backbone.patch_embed1", self.backbone.patch_embed1)
        # print("out1", out_1.shape)
        # print("self.backbone.block1", self.backbone.block1)
        for i, blk in enumerate(self.backbone.block1):
            out_1 = blk(out_1, H, W)
        out_1 = self.backbone.norm1(out_1)
        # print("out1", out_1.shape)
        out_1 = out_1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[0], 88, 88)
        # print("out1", out_1.shape)

        # stage 2
        out_2, H, W = self.backbone.patch_embed2(out_1)
        # print("self.backbone.patch_embed2", self.backbone.patch_embed2)
        # print("out_2", out_2.shape)
        # print("self.backbone.block2", self.backbone.block2)
        for i, blk in enumerate(self.backbone.block2):
            out_2 = blk(out_2, H, W)
        out_2 = self.backbone.norm2(out_2)
        # print("out_2", out_2.shape)
        out_2 = out_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[1], 44, 44)
        # print("out_2", out_2.shape)

        # stage 3
        out_3, H, W = self.backbone.patch_embed3(out_2)
        # print("self.backbone.block3", self.backbone.block3)
        # print("out_3", out_3.shape)
        for i, blk in enumerate(self.backbone.block3):
            out_3 = blk(out_3, H, W)
        out_3 = self.backbone.norm3(out_3)
        # print("out_3", out_3.shape)
        out_3 = out_3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[2], 22, 22)
        # print("out_3", out_3.shape)

        # stage 4
        out_4, H, W = self.backbone.patch_embed4(out_3)
        for i, blk in enumerate(self.backbone.block4):
            out_4 = blk(out_4, H, W)
        out_4 = self.backbone.norm4(out_4)
        out_4 = out_4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  #(Batch_Size, self.backbone.embed_dims[3], 11, 11)

        #segmentation
        # go through LP Header
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # linear fuse and go pass LP Header
        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([lp_3, F.interpolate(lp_4,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([lp_2, F.interpolate(lp_34,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))
        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([lp_1, F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)], dim=1)))

        # get the final output
        lp4_resized = F.interpolate(lp_4,scale_factor=8,mode='bilinear', align_corners=False)
        lp3_resized = F.interpolate(lp_34,scale_factor=4,mode='bilinear', align_corners=False)
        lp2_resized = F.interpolate(lp_23,scale_factor=2,mode='bilinear', align_corners=False)
        lp1_resized = lp_12

        out1 = self.linear_pred(torch.cat([lp1_resized, lp2_resized, lp3_resized, lp4_resized], dim=1))
        # print(out.shape)

        #classification
        out2 = self.global_avg_pool(out_4)
        out2 = self.norm1(out2)
        out2 = self.Relu(out2)
        out2 = self.Dropout(out2)
        out2 = self.conv1(out2)
        out2 = self.norm2(out2)
        out2 = self.Relu(out2)
        out2 = self.conv2(out2)

        return out1, out2

from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision, MultilabelConfusionMatrix
import scipy.sparse.csgraph._laplacian

class_names = ["specularity", "saturation", "artifact", "blur", "contrast", "bubbles", "instrument", "blood"]

class_weights = {
    'specularity': 0.10,
    'saturation': 0.1,#0.10,
    'artifact': 0.1,#0.50,
    'blur': 0.1,#0.05,
    'contrast': 0.1,#0.05,
    'bubbles': 0.3, #0.10,
    'instrument': 0,  # Add 0 or relevant weight if necessary
    'blood': 0  # Add 0 or relevant weight if necessary
}
location_weights = {
    "center": 0.25,
    "left": 0.25,
    "right": 0.25,
    "top": 0.25,
    "bottom": 0.25,
    "top-left": 0.125,
    "top-right": 0.125,
    "bottom-left": 0.125,
    "bottom-right": 0.125
}

def iou(boxA, boxB):
    # Compute the intersection area
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute the IoU
    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value

def filter_boxes(boxes, labels, iou_threshold=0.4):
    # Filter out boxes based on IoU threshold, always keep label == 5
    remove = set()  # Use a set to track indices to discard
    for i in range(len(boxes)):
        if i in remove:
            continue
        if labels[i] == 5:
            continue
        for j in range(i + 1, len(boxes)):
            if j in remove:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                remove.add(j)  # Mark j for removal if IoU exceeds threshold

    keep = [i for i in range(len(boxes)) if i not in remove]
    return [boxes[i] for i in keep], [labels[i] for i in keep]


def compute_qs(artifacts, total_image_area):
    total_score = 0
    for artifact in artifacts:
        wc = class_weights.get(artifact['class_name'], 0)
        wa = artifact['area'] / total_image_area
        wl = artifact['location_weight']
        total_score += 0.5 * wc * wa + 0.5 * wc * wl

    qs = 1 - total_score
    return qs

def get_location_weight(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    
    if x_center < img_width / 3:
        if y_center < img_height / 3:
            return location_weights['top-left']
        elif y_center > 2 * img_height / 3:
            return location_weights['bottom-left']
        else:
            return location_weights['left']
    elif x_center > 2 * img_width / 3:
        if y_center < img_height / 3:
            return location_weights['top-right']
        elif y_center > 2 * img_height / 3:
            return location_weights['bottom-right']
        else:
            return location_weights['right']
    else:
        if y_center < img_height / 3:
            return location_weights['top']
        elif y_center > 2 * img_height / 3:
            return location_weights['bottom']
        else:
            return location_weights['center']



def saveResult():

    ESFPNet = torch.load(args.saved_model)
    ESFPNet.eval()

    model = torch.load('/home/dunpt1504/Documents/School_project/Quality_assessment/faster_rcnn_resnet50_fpn_custom_2500_roboflow_483.pth')
    model.eval()
    model.to(device)


    total = 0
    num_class = 8
    total_correct_predictions = torch.zeros(num_class).to(device)
    threshold_class = 0.6
    alpha = 1

    shape = (num_class, 2, 2)
    metric_all_val = torch.zeros(shape).to(device)

    metric = MultilabelConfusionMatrix(num_labels=num_class, threshold= threshold_class)
    metric = metric.to(device)

    label_list = ['Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina', 'Mucosal infiltration', 'Vascular growth', 'Tumor', 'Benign']

    val_loader = test_dataset(args.path_imgs_test + '/',args.path_masks_test + '/', args.label_json_path , args.init_trainsize) #
    
    benign_list = []
    uncertain_list = []
    lesion_list = []

    for i in range(val_loader.size):
        image, labels_tensor, image_qs, name = val_loader.load_data()#
        # print(name)
        image = image.cuda()
        image_qs = image_qs.cuda()
        labels_tensor = labels_tensor.to(device)

        with torch.no_grad():
            prediction = model(image_qs)
        
        boxes = prediction[0]['boxes'].cpu().numpy()
        # print(boxes)
        labels = prediction[0]['labels'].cpu().numpy()

        # boxes, labels = filter_boxes(boxes, labels, iou_threshold=0.4)

        if os.path.exists('/home/dunpt1504/Documents/School_project/Uncertainty/test_dataset_qs/imgs/' + name + '.png'):
            img = Image.open('/home/dunpt1504/Documents/School_project/Uncertainty/test_dataset_qs/imgs/' + name + '.png').convert("RGB")
        else:
            img = Image.open('/home/dunpt1504/Documents/School_project/Uncertainty/test_dataset_qs/imgs/' + name + '.jpg').convert("RGB")
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _ , _, img_height, img_width = image.shape
        artifacts = []

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        class_names = ["specularity", "saturation", "artifact", "blur", "contrast", "bubbles", "instrument", "blood"]
        for box, label in zip(boxes, labels):
            if label == 5:
                # print(prediction)
                x_min, y_min, x_max, y_max = map(int, box)

                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

                # class_name = class_names[label]
                # cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # cv2.putText(img_cv, f'{class_name}', (x_min, y_min + 10), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

               
        # print('/home/dunpt1504/Documents/School_Project/Uncertainty/infered/od/' + name + '.jpg')


        # cv2.imwrite('/home/dunpt1504/Documents/School_project/Uncertainty/infered/od/' + name + '.jpg', img_cv)
        

        # cv2.imwrite('/home/dunp/t1504/Documents/School_project/Uncertainty/infered/od_mask/' + name + '.jpg', mask)

        pred1, pred2 = ESFPNet(image)

        pred1 = F.upsample(pred1, size=img_height, mode='bilinear', align_corners=False)
        pred1 = pred1.sigmoid()
        threshold = torch.tensor([0.5]).to(device)
        pred1 = (pred1 > threshold).float() * 1

        pred1 = pred1.data.cpu().numpy().squeeze()
        pred1 = (pred1 - pred1.min()) / (pred1.max() - pred1.min() + 1e-8)

        

        intersection = np.logical_and(pred1, mask).sum()
        union = np.logical_or(pred1, mask).sum()

        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        # print("IoU:", iou)

        if name.startswith("CH"):
            benign_list.append([name, iou])  
        elif name.startswith("uncertainty"):
            # print(prediction)
            uncertain_list.append([name, iou])
        else:
            lesion_list.append([name, iou])

        
        # if (1 - iou) <= 0.76:
        #     print(name)
        #     imageio.imwrite('/home/dunpt1504/Documents/School_project/Uncertainty/infered/segment_mask/' + name + '.jpg', img_as_ubyte(pred1))

        # if iou < 0.2266:#:0.2373
        pred2 = np.squeeze(pred2)
        pred2 = torch.unsqueeze(pred2, 0)
        
        total += 1


        labels_predicted = torch.sigmoid(pred2)
        # print(name, labels_predicted)
        thresholded_predictions = (labels_predicted >= threshold_class).int()
        correct_predictions = (thresholded_predictions == labels_tensor).sum(dim=0)
        # print(correct_predictions)
        total_correct_predictions += correct_predictions

        target_tensor = torch.unsqueeze(labels_tensor, 0)
        # if name.startswith("uncertainty"):
        #     print(labels_predicted)
        
        # predicted_data = [label_list[i] for i, value in enumerate(thresholded_predictions.squeeze().tolist()) if value == 1]
        
        # ground_truth = [label_list[i] for i, value in enumerate(target_tensor.squeeze().tolist()) if value == 1]
        # if name.startswith("uncertainty"):
        #     print(name, " ",predicted_data)      
        # with open(path, "a") as file:
        #     file.write(file_name + "\n")
        #     file.write("Predict:" + str(predicted_data) + "\n")
        #     file.write("Ground truth:" + str(ground_truth) + "\n" + "\n")

        cm = metric(labels_predicted, target_tensor.int())
        # if not name.startswith("uncertainty") or not name.startswith("CH"):
        metric_all_val += cm
        # else:
        #     labels_predicted = torch.tensor([[0.1254, 0.0218, 0.1622, 0.3328, 0.0413, 0.0603, 0.0705, 0.7601]]).to(device)
        #     cm = metric(labels_predicted, target_tensor.int())
        #     metric_all_val += cm
            
        #     total += 1
        #     thresholded_predictions = (labels_predicted >= threshold_class).int()
        #     correct_predictions = (thresholded_predictions == labels_tensor).sum(dim=0)
        #     # print(correct_predictions)
        #     total_correct_predictions += correct_predictions

        
            


    

    confusion_matrix = metric_all_val
    # print(confusion_matrix)
    
    precision_macro = 0
    recall_macro = 0
    TP_all = 0
    FP_all = 0
    FN_all = 0
    for i in range(num_class):
        TN = confusion_matrix[i][0][0]
        FP = confusion_matrix[i][0][1]
        FN = confusion_matrix[i][1][0]
        TP = confusion_matrix[i][1][1]
        TP_all += TP
        FP_all += FP
        FN_all += FN
        precision = TP/(TP+FP)
        precision_macro += precision
        recall = TP/(TP+FN)
        recall_macro += recall
        print(label_list[i] + " Precision ", float(precision.item()))
        print(label_list[i] + " Recall ", float(recall.item()))
    
    recall_micro = TP_all/(TP_all+FN_all)
    recall_macro = recall_macro/num_class
    precision_micro = TP_all/(TP_all+FP_all)
    precision_macro = precision_macro/num_class
    F1_micro = 2*recall_micro*precision_micro/(recall_micro+precision_micro)
    F1_macro = 2*recall_macro*precision_macro/(recall_macro+precision_macro)
    print("Recall (micro)", 100 * float(recall_micro.item()))
    print("Recall (macro)", 100 * float(recall_macro.item()))
    print("Precision (micro)", 100 * float(precision_micro.item()))
    print("Precision (macro)", 100 * float(precision_macro.item()))
    print("F1 (micro)", 100 * float(F1_micro.item()))
    print("F1 (macro)", 100 * float(F1_macro.item()))
    overall_accuracy = torch.mean(total_correct_predictions) / total
    print("acc_val_classification", 100 * overall_accuracy.item())

    # confusion_matrix_save = metric_all_val.cpu().numpy()
    # for i in range(num_class):
    #     plt.figure(figsize=(5, 5))
    #     sns.heatmap(confusion_matrix_save[i], annot=True, fmt=".2f", cmap="Blues", cbar=False)
    #     plt.title(f"Confusion Matrix - {label_list[i]}")
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     path = f"/home/dunpt1504/Documents/School_project/Uncertainty/cf_matrix/test_dataset_no_uncertainty/confusion_matrix_{label_list[i].replace(' ', '_')}.png"
    #     os.makedirs(f"/home/dunpt1504/Documents/School_project/Uncertainty/cf_matrix/test_dataset_no_uncertainty", exist_ok=True)
    #     plt.savefig(path)
    #     plt.close()

    
    # text_path = '/home/dunpt1504/Documents/School_project/Quality_assessment/IOU_scores/test_dataset_qs_EAD2019'  # Folder to save inferred images
    # os.makedirs(text_path, exist_ok=True)
    # with open(text_path + '/Benign.txt', "w") as file:
    #     for item in benign_list:
    #         file.write(f"{item}\n")

    # with open(text_path + '/Uncertain.txt', "w") as file:
    #     for item in uncertain_list:
    #         file.write(f"{item}\n")

    # with open(text_path + '/Lesions.txt', "w") as file:
    #     for item in lesion_list:
    #         file.write(f"{item}\n")


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Models moved to GPU.')
else:
    print('Only CPU available.')

saveResult()