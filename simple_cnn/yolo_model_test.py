import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("IMG_1482.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ 프레임 리사이즈 (예: 960 x 540)
    resized_frame = cv2.resize(frame, (960, 540))

    # ✅ 모든 프레임에 대해 추론
    results = model.predict(source=resized_frame, imgsz=416, conf=0.25, stream=True)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # 바운딩 박스 그리기
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("YOLOv8 Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
