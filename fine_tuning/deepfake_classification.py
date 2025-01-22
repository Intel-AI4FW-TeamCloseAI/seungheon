import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 작업 디렉토리

# 1) device 설정 (GPU 사용 가능 시 GPU 할당)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) 모델 구조 정의 (학습 때와 동일하게)
test_model_path = os.path.join(base_dir, "efficientnet_b0_deepfake_s128.pt")
model = torch.jit.load(test_model_path, map_location=device)
model.eval()

# 4) 이미지 전처리 transform (학습 때와 동일하게)
img_resized = (128, 128)

transform = transforms.Compose([
    transforms.Resize(img_resized),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 5) 예측할 이미지 폴더 경로
image_folder = os.path.join(base_dir, 'sample')

# 6) 폴더 내 이미지들에 대해 일괄 추론
with torch.no_grad():
    for filename in os.listdir(image_folder):
        # 파일 확장자 확인 (jpg, png 등)
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(image_folder, filename)

            # 이미지 열기
            img = Image.open(img_path).convert('RGB')
            # 전처리
            img_tensor = transform(img).unsqueeze(0).to(device)  # shape: (1, C, H, W)

            # 모델 추론
            outputs = model(img_tensor)
            # 예: CrossEntropyLoss 기준 => argmax
            _, pred = torch.max(outputs, 1)

            # 0 -> Fake, 1 -> Real (ImageFolder 학습 기준)
            label_idx = pred.item()
            label_str = "AIGenerated" if label_idx == 0 else ("Fake" if label_idx == 1 else "Real")

            print(f"[{filename}] => {label_str}")
