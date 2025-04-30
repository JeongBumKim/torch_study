import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

# 클래스
classes = ["__background__", "sirbot"]

transform = A.Compose([
    A.OneOf([
        A.Resize(320, 320),
        A.Resize(416, 416),
        A.Resize(512, 512),
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=1.0),
    ], p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))



# Augmented Dataset 정의
class YoloDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, classes, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        boxes, labels = [], []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts[:5])
                xmin = (x_center - w / 2) * width
                ymin = (y_center - h / 2) * height
                xmax = (x_center + w / 2) * width
                ymax = (y_center + h / 2) * height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id) + 1)

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
            image = transformed['image'].float() / 255.0  # ← 🔥 오류 방지 핵심

            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['category_ids'], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target

    def __len__(self):
        return len(self.image_files)

# DataLoader 교체
dataset = YoloDatasetAugmented("train/images", "train/labels", classes, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 모델
model = get_model(num_classes=len(classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# 옵티마이저
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 학습
model.train()
for epoch in range(30):
    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

    
torch.save(model.state_dict(), "faster_rcnn_cup2.pth")
print("✅ 모델이 저장되었습니다: faster_rcnn_cup2.pth")