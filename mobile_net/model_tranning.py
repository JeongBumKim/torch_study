import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

NUM_CLASSES = 2  # ← 클래스 수를 여기에 맞게 설정하세요
SAVE_PATH = "mobilenetv2_classifier.pth"

# 📦 사용자 정의 데이터셋 클래스
class YoloStyleImageDataset(Dataset):
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
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            class_id = int(f.readline().strip().split()[0])  # 첫 번째 클래스 ID만 사용

        return image, class_id

# 🧪 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 🧷 DataLoader
train_dataset = YoloStyleImageDataset('dataset/train/images', 'dataset/train/labels', transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 🧠 MobileNetV2 기반 모델
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🛠️ 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔁 학습 루프
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}")

# 💾 모델 저장
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ 모델 저장 완료: {SAVE_PATH}")
