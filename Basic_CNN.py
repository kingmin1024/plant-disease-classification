from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 크기
img_size = (128, 128)

# 배치 크기
batch_size = 16

# 데이터 경로
train_path = "Dataset_split/train"
val_path = "Dataset_split/val"
test_path = "Dataset_split/test"

# 데이터 증강 + 정규화
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# 검증/테스트는 정규화만
val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# train 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# validation 데이터 로드
val_generator = val_test_datagen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# test 데이터 로드
test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

# CNN 모델 생성
model = Sequential()

# 첫 번째 CNN 층
model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(128,128,3)
))

model.add(MaxPooling2D(pool_size=(2,2)))

# 두 번째 CNN 층
model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(2,2)))

# 평탄화
model.add(Flatten())

# Dense 층
model.add(Dense(128, activation='relu'))

# 과적합 방지
model.add(Dropout(0.5))

# 출력층
model.add(Dense(23, activation='softmax'))

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 모델 학습
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

import matplotlib.pyplot as plt

# Accuracy 그래프
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

# Loss 그래프
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Accuracy :", test_accuracy)

model.save("plant_disease_cnn_model.h5")

print("모델 저장 완료!")