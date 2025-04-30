import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# ✅ 1. 데이터셋 불러오기
transform = transforms.Compose([
    transforms.ToTensor(),                  # 이미지를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5),    # 평균 0.5, 표준편차 0.5로 정규화
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

# ✅ 2. 간단한 CNN 모델 만들기
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ 3. 모델, 손실 함수, 옵티마이저 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
print(device)
criterion = nn.CrossEntropyLoss()  # 분류 문제용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ 4. 학습 루프
for epoch in range(10):  # 10 epochs
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()         # 그라디언트 초기화
        outputs = model(images)        # 순전파
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()               # 역전파
        optimizer.step()              # 가중치 업데이트

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

print('Finished Training')

# ✅ 5. 테스트 루프
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
torch.save(model.state_dict(), 'cnn_model.pth')
