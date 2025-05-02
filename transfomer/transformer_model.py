import os
import yaml
from ultralytics import YOLO

# ✅ 클래스 정의
class_names = ["sirbot"]  # 클래스 이름은 필요에 따라 수정

# ✅ 절대 경로로 지정
root_dir = "/home/rgt/jb_ws/torch_study/transfomer/data/train"
image_dir = os.path.join(root_dir, "images")
label_dir = os.path.join(root_dir, "labels")
data_yaml_path = '/home/rgt/jb_ws/torch_study/transfomer/data/train/data.yaml'
print(image_dir)

# ✅ 1. data.yaml 생성
data_yaml = {
    "train": image_dir,
    "val": image_dir,  # val 없이 train만 사용
    "nc": len(class_names),
    "names": class_names
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

# ✅ 2. YOLOv8 모델 로드 및 학습
model = YOLO("yolov8n.pt")  # 또는 yolov8s.pt, yolov8m.pt

model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,       # GPU (또는 'cpu')
    val=False       # 검증 생략
)
