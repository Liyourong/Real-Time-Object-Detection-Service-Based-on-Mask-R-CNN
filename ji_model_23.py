from pycocotools.coco import COCO
import os

# Update the data directory
data_dir = '/content/datasets'

# Path to the annotation file (update this if the file name is different)
annotations_path = os.path.join(data_dir, 'annotations/annotations/instances_train2017.json')

# Initialize COCO API
try:
    coco = COCO(annotations_path)
except FileNotFoundError:
    print(f"Annotation file not found at {annotations_path}")

import requests
import os
import zipfile
import torch
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision
from torchvision.models.detection import FasterRCNN

# Load pre-trained ResNet-50 model as baseline
model = torchvision.models.resnet50(pretrained=True)

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

# Initialize the ResNet model
model = resnet50(pretrained=True)

# Define the number of classes (COCO has 80 classes + 1 for background)
num_classes = 81

# Get the number of input features for the classifier
in_features = model.fc.in_features

# Replace the fully connected layer with a custom classifier
model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, num_classes)
)

# Define the device to train on (CPU or GPU)
device = torch.device('cuda')
model.to(device)

# Define the optimizer (only optimize the parameters of the custom classifier)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Define transform for testing data
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_dataset = CocoDetection(root='/content/datasets/train2017/train2017',
                              annFile='/content/datasets/annotations/annotations/instances_train2017.json',
                              transform=transform)

from torch.utils.data.dataloader import default_collate

def custom_collate(batch):
    images = [item[0] for item in batch]
    all_targets = [item[1] for item in batch]

    # Flatten and standardize labels
    labels = []
    for targets in all_targets:
        if isinstance(targets, list) and len(targets) > 0 and 'labels' in targets[0]:
            # For simplicity, let's take the first label of the first annotation per image
            labels.append(targets[0]['labels'][0])
        else:
            labels.append(23)  # Placeholder for images without annotations

    images = default_collate(images)
    labels = default_collate(labels)
    return images, labels

data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=custom_collate)

import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# Load the trained model
model.load_state_dict(torch.load('trained_classifier.pth'))
model.eval()  # Set the model to evaluation mode

# Load the image
image = Image.open('/content/jijijijiji.jpeg')

# Define the transform (this should be the same as used during training)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the image
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Preprocess the image
image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
image_tensor = image_tensor.to('cuda')

# Make predictions
with torch.no_grad():
    output = model(image_tensor)

# Process the output to get the predicted annotations
# (This step depends on what your model is predicting)

# Example: For a simple classification model
_, predicted_class = torch.max(output, 1)
print(f"Predicted class: {predicted_class.item()}")

# Assuming 'output' contains both class labels and bounding boxes
# The exact structure depends on your model

# Example: Output might be a list of dictionaries with keys 'boxes' and 'labels'
for prediction in output:
    boxes = prediction['boxes']
    labels = prediction['labels']

    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        # Draw the bounding box
        # Box format: [x_min, y_min, x_max, y_max]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")

        # Optionally, add text for the label
        draw.text((box[0], box[1]), f"Class: {label}")

image.show()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        # Code is hide

    def __getitem__(self, idx):
        # Code is hide

        return img, target

    def __len__(self):
        return len(self.imgs)

def custom_maskrcnn_resnet50_fpn(weights="imagenet", pretrained_backbone=True):
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)

    model = detection_models.MaskRCNN(backbone, num_classes=91)

    if weights == "imagenet":
        state_dict = detection_models.maskrcnn_resnet50_fpn(pretrained=True).state_dict()
        model.load_state_dict(state_dict)

    return model

class CustomPredictor(nn.Module):
    def __init__(self, in_features, num_classes):
        # Code is hide

    def forward(self, x):
        # Code is hide

class CustomMaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_layer, num_classes):
        # Code is hide

    def forward(self, x):
        # Code is hide

def get_model_instance_segmentation(num_classes):
    # Code is hide

import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def show_out(image):
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]

    pred_labels = [f"{100*score:.3f}% Human!!!" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))

# Code is hide for training, image testing, and showing for academic integrity intention
