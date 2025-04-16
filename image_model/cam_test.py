import cv2

# 0번 카메라(기본 웹캠) 열기
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("📷 웹캠이 열렸습니다. 'q' 키를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
