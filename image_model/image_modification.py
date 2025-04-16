import cv2

import cv2
import os

# 이미지가 저장된 폴더 경로
image_dir = "image_model/image"  # 여기에 이미지들이 저장돼 있어야 함
save_dir = "image_model/image"
os.makedirs(save_dir, exist_ok=True)

# 이미지 파일 이름 리스트
image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
print(f"🔍 {len(image_names)}개의 이미지가 준비되었습니다.")

for idx, img_name in enumerate(image_names):
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 이미지 로딩 실패: {img_name}")
        continue

    clone = image.copy()

    print(f"\n[{idx+1}/{len(image_names)}] 🔧 {img_name} - 자를 영역을 선택하고 Enter, 취소하려면 c")

    # ROI 선택
    roi = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi

    if w > 0 and h > 0:
        cropped = clone[y:y+h, x:x+w]
        save_path = os.path.join(save_dir, f"cropped_{img_name}")
        cv2.imwrite(save_path, cropped)
        print(f"✅ 저장 완료: {save_path}")
    else:
        print("⚠️ ROI 선택이 취소되었습니다.")

    cv2.destroyWindow("Image")

print("\n🎉 모든 이미지 처리가 완료")
