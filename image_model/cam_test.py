import cv2

cap = cv2.VideoCapture(0)  # 기본 웹캠. 외부 카메라면 1, 2 등으로 바꿔보세요

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ 카메라 테스트 시작. ESC 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("Camera Test", frame)

    # ESC 키 (27) 입력 시 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
