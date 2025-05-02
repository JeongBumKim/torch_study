import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import torchvision.transforms.functional as F

import torch.nn as nn
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
# ✅ 클래스 설정
CLASS_NAMES = ["cup"]
NUM_CLASSES = 2  # 클래스 1개 + 배경

# ✅ YOLO TXT → BBox 변환 함수
def yolo_to_bbox(label_path, img_width, img_height):
    boxes = []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id + 1)  # class_id + 1 (0은 background)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

# ✅ 데이터셋 정의
class YoloDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        h, w = image.shape[:2]

        boxes, labels = yolo_to_bbox(label_path, w, h)

        image_tensor = F.to_tensor(image)
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image_tensor, target

# ✅ 데이터 로더
dataset = YoloDetectionDataset("dataset/train/images", "dataset/train/labels")
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ✅ 모델 불러오기 및 분류기 수정
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)

dummy_input = torch.randn(1, 3, 320, 320)
features = model.backbone(dummy_input)
in_channels = [f.shape[1] for f in features]  # 예: [672, 160, 96, 96, 64, 64]

# ✅ anchor 수는 리스트로 반환됨
num_anchors = model.anchor_generator.num_anchors_per_location()  # 예: [6, 6, 6, 6, 6, 6]

# ✅ classification head 재정의 (이제 리스트끼리 정확히 대응됨)
model.head.classification_head = SSDLiteClassificationHead(
    in_channels=[672, 160, 96, 96, 64, 64],  # or 자동 추출값
    num_anchors=[6, 6, 6, 6, 6, 6],
    num_classes=NUM_CLASSES,
    norm_layer=nn.BatchNorm2d
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 옵티마이저 설정
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# ✅ 학습 루프
model.train()
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

# ✅ 모델 저장
torch.save(model.state_dict(), "ssdlite_cup_detector.pth")
print("✅ 모델 저장 완료: ssdlite_cup_detector.pth")
