from ultralytics import YOLO

# ✅ 모델 로드
model = YOLO("yolo11n.pt")  # 또는 'runs/detect/train/weights/best.pt'

# ✅ 테스트 이미지 추론
results = model("test.jpg", show=True)  # show=True → 이미지 창 열림 (로컬에서 실행 시)

# ✅ 바운딩 박스 정보 출력
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    print("Detected:", len(boxes), "objects")
