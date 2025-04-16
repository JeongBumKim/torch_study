import cv2

# 1. 웹캠 열기 (기본 카메라 = 0)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

img_count = 0

print("📸 's' 키를 누르면 이미지 저장, 'q' 키를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 가져올 수 없습니다.")
        break

    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. 화면에 출력
    cv2.imshow("Grayscale Camera", gray)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"image_model/captured_image_{img_count}.png"
        cv2.imwrite(filename, gray)
        print(f"✅ 저장됨: {filename}")
        img_count += 1

    elif key == ord('q'):
        print("👋 종료합니다.")
        break

# 4. 자원 정리
cap.release()
cv2.destroyAllWindows()
