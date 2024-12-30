import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from PIL import Image
import os

import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import wandb 
wandb.login()
wandb.init(project="Fast RCNN")


class CustomDataset(Dataset):
    def __init__(self, frames_dir, gtbox_dir, save_dir, transforms=None):
        self.frames_dir = frames_dir
        self.gtbox_dir = gtbox_dir
        self.save_dir = save_dir  # Directory to save images with boxes
        os.makedirs(save_dir, exist_ok=True)  # Create save directory if it doesn't exist
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(frames_dir)))
        self.annotations = list(sorted(os.listdir(gtbox_dir)))

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.frames_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Load ground truth boxes
        gtbox_path = os.path.join(self.gtbox_dir, self.annotations[idx])
        # print(gtbox_path)
        boxes, labels = self.parse_gtbox(gtbox_path, img_path, idx)    

        # Create target dictionary
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)  # Bounding boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)  # Class labels

        if self.transforms is not None:
            img = self.transforms(img)

        # Save image with bounding boxes drawn
        # self.save_image_with_boxes(img_path, boxes, labels, idx)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def parse_gtbox(self, gtbox_path, img_path, idx):
        """Helper function to parse gtbox files and return bounding boxes and labels."""
        height, width = cv2.imread(img_path).shape[:2]
        boxes = []
        labels = []
        with open(gtbox_path) as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
                
                # Convert normalized coordinates (x_center, y_center, width, height)
                # to (x_min, y_min, x_max, y_max)
                x_min = (x_center - box_width / 2) * width
                y_min = (y_center - box_height / 2) * height
                x_max = (x_center + box_width / 2) * width
                y_max = (y_center + box_height / 2) * height
                
                # boxes.append([x_min, y_min, x_max, y_max])
                # labels.append(label)

                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)
                # else:
                #     print(f"Warning: Invalid bounding box found for image {img_path} with box [{x_min}, {y_min}, {x_max}, {y_max}]")

        # print(labels)
        return boxes, labels

    def save_image_with_boxes(self, img_path, boxes, labels, idx):
        """Helper function to draw bounding boxes and save the image to a folder."""
        img = cv2.imread(img_path)
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = map(int, box)
            # Draw rectangle (BGR for OpenCV)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Put label text
            cv2.putText(img, f'Label: {label}', (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Save the image with bounding boxes to the save directory
        save_path = os.path.join(self.save_dir, f'image_with_boxes_{idx}.jpg')
        cv2.imwrite(save_path, img)



# Data augmentations (if needed)
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())  # Convert image to tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Random horizontal flip
    return T.Compose(transforms)


# Load the dataset
frames_dir = "/home/ailab/Documents/dunpt/Quality_assessment/train_roboflow_483/images"
gtbox_dir = "/home/ailab/Documents/dunpt/Quality_assessment/train_roboflow_483/labels"
save_dir = '/home/ailab/Documents/dunpt/Quality_assessment/train_roboflow_483/save_img'
train_dataset = CustomDataset(frames_dir, gtbox_dir, save_dir,transforms=get_transform(train=True))
# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

#new
# Load pre-trained Faster R-CNN model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# num_classes = 9  # For example, 5 object classes + 1 background
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#continue
saved_model_path = 'faster_rcnn_resnet50_fpn_599_bubbles_EAD2019.pth'
model = torch.load(saved_model_path)

# Move model to device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Custom Training Loop
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Loop over the batches
    for images, targets in tqdm(train_loader):
        # Move images and targets to the correct device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Print average loss for this epoch
    torch.save(model, 'faster_rcnn_resnet50_fpn_custom_2500_roboflow_483.pth')
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")
    wandb.log({"Loss_train" : total_loss/len(train_loader)})

# Save the trained model
# torch.save(model, 'faster_rcnn_resnet50_fpn_custom.pth')
wandb.finish()