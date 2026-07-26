# =========================
# 1. 라이브러리 import
# =========================

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt

# =========================
# 2. 데이터 경로 설정
# =========================

train_path = "Dataset_split/train"
val_path = "Dataset_split/val"
test_path = "Dataset_split/test"

# =========================
# 3. 이미지 설정
# =========================

img_size = (128, 128)
batch_size = 16
# =========================
# 4. 데이터 정규화
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================
# 5. 데이터 로드
# =========================

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 6. MobileNetV2 모델 생성
# =========================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

# 기존 가중치 고정
base_model.trainable = False

# 모델 생성
model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

# 출력층
model.add(Dense(23, activation='softmax'))

# =========================
# 7. 모델 컴파일
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# =========================
# 8. 모델 학습
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# =========================
# 9. 테스트 평가
# =========================

test_loss, test_accuracy = model.evaluate(test_generator)

print("MobileNetV2 Test Accuracy :", test_accuracy)

# =========================
# 10. Accuracy 그래프
# =========================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('MobileNetV2 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

# =========================
# 11. Loss 그래프
# =========================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('MobileNetV2 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

import numpy as np
from tensorflow.keras.preprocessing import image

# 테스트 이미지 경로
img_path = "sa/test.jpg"

# 이미지 로드
img = image.load_img(
    img_path,
    target_size=(128,128)
)

# 이미지 배열 변환
img_array = image.img_to_array(img)

# 정규화
img_array = img_array / 255.0

# 차원 추가
img_array = np.expand_dims(img_array, axis=0)

# 예측
prediction = model.predict(img_array)

# 가장 높은 확률 클래스 인덱스
predicted_class_index = np.argmax(prediction)

# 클래스 이름 가져오기
class_names = list(train_generator.class_indices.keys())

# 예측 클래스 이름
predicted_class_name = class_names[predicted_class_index]

# 신뢰도
confidence = np.max(prediction)

# 결과 출력
print("예측 결과 :", predicted_class_name)
print("신뢰도 :", confidence)

# 이미지 출력
plt.imshow(img)
plt.title(f"Prediction : {predicted_class_name}")
plt.axis("off")
plt.show()