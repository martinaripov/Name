from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import os
import json

DATASET_PATH = 'dataset/'

num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'

train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,     # 20% картинок пойдут в валидацию
    rotation_range=25,        # случайные повороты до 25 градусов
    width_shift_range=0.2,    # сдвиги по ширине
    height_shift_range=0.2,   # сдвиги по высоте
    shear_range=0.2,          # "срез" изображения
    zoom_range=0.2,           # приближение/отдаление
    horizontal_flip=True,     # отражение по горизонтали
    fill_mode='nearest'       # как заполнять пустые пиксели
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='validation'
)

with open('class_indices.json', 'w') as f:
    json.dump(train_data.class_indices, f)


model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid') if class_mode == 'binary' else Dense(num_classes, activation='softmax')
])

loss_function = 'binary_crossentropy' if class_mode == 'binary' else 'categorical_crossentropy'
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stop])

test_loss, test_accuracy = model.evaluate(val_data)
print(f'Точность модели на валидационных данных: {test_accuracy:.2f}')

model.save('image_classifier.keras')