import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ 클래스 리스트
classes = ["__background__", "cup"]

# ✅ Dataset 클래스
class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, classes, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

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

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.image_files)

# ✅ 모델 생성 함수
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = YoloDataset("train/images", "train/labels", classes, transforms=T.ToTensor())
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = get_model(num_classes=len(classes))
model.to(device)

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=0.005, momentum=0.9, weight_decay=0.0005)

# ✅ 학습
model.train()
for epoch in range(10):
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"📚 Epoch {epoch+1} | Loss: {losses.item():.4f}")

# ✅ 저장
torch.save(model.state_dict(), "faster_rcnn_cup.pth")
print("✅ 모델 저장 완료: faster_rcnn_cup.pth")

# ✅ 추론 테스트 (1장 예시)
model.eval()
test_img_path = "test.jpg"
image = Image.open(test_img_path).convert("RGB")
img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)[0]

# ✅ 시각화
fig, ax = plt.subplots(1)
ax.imshow(image)

for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
    if score > 0.5:
        x1, y1, x2, y2 = [int(v.item()) for v in box]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(x1, y1 - 10, f"{classes[label]} {score:.2f}", color='red')

plt.axis("off")
plt.show()
