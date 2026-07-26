import matplotlib.pyplot as plt
import cv2
import os
import random

dataset_path = "Dataset"

# 랜덤 클래스 선택
folder = random.choice(os.listdir(dataset_path))

# 랜덤 이미지 선택
img_name = random.choice(
    os.listdir(os.path.join(dataset_path, folder))
)

# 이미지 경로
img_path = os.path.join(dataset_path, folder, img_name)

# 이미지 읽기
img = cv2.imread(img_path)

# BGR → RGB 변환
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 출력
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(folder)
plt.axis("off")
plt.show()