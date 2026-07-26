import os
import shutil
import random

# 원본 데이터셋 경로
dataset_path = "Dataset"

# 새로 생성할 폴더
output_path = "Dataset_split"

# 비율 설정
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# train / val / test 폴더 생성
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

# 클래스별 처리
for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):

        images = os.listdir(class_path)

        # 이미지 섞기
        random.shuffle(images)

        total = len(images)

        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # 각 split 폴더 생성
        for split in ['train', 'val', 'test']:
            os.makedirs(
                os.path.join(output_path, split, class_name),
                exist_ok=True
            )

        # 파일 복사 함수
        def copy_files(image_list, split_name):
            for image in image_list:

                src = os.path.join(class_path, image)

                dst = os.path.join(
                    output_path,
                    split_name,
                    class_name,
                    image
                )

                shutil.copy(src, dst)

        # 복사 실행
        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        copy_files(test_images, 'test')

print("데이터 분할 완료!")