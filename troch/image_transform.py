from PIL import Image
from torchvision import transforms

# 이미지 열기
image = Image.open("boots.jpg").convert("RGB")

# 변환 정의
transform = transforms.ToTensor()

# 이미지 → 텐서로 변환
tensor_image = transform(image)

print(tensor_image.shape)  # 예: [3, H, W]

from PIL import Image
import numpy as np
import torch


# numpy 배열로 변환 → 텐서로 변환
np_image = np.array(image).astype(np.float32) / 255.0  # [H, W, C]
tensor_image = torch.from_numpy(np_image).permute(2, 0, 1)  # [C, H, W]

print(tensor_image[:, 120, 120])
