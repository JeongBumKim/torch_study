import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("사용 중 디바이스:", device)

# CIFAR-10 데이터셋
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기 맞춤
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# 모델 (ResNet18)
model = models.resnet18(pretrained=False, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 시작
start_total = time.time()
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_epoch = time.time()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    elapsed = time.time() - start_epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Time: {elapsed:.2f} sec")

total = time.time() - start_total
print(f"\n📊 총 학습 시간: {total:.2f}초 (device: {device})")
