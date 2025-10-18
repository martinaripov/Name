from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import json

DATASET_PATH = 'dataset/'
with open('class_indices.json') as f:
    class_indices = json.load(f)
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'
class_names = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'Ошибка: файл не найден по пути: {image_path}')
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f'Ошибка: Поврежденное изображение - {image_path}')
        return
    model = tf.keras.models.load_model('image_classifier.keras')

    img = cv2.imread(image_path)
    if img is None:
        print(f'Ошибка: Не удалось прочитать изображение - {image_path}')
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RG)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if class_mode == 'binary':
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
    else:
        predicted_class = class_names[int(tf.argmax(prediction, axis=-1).numpy()[0])]

    print(f'Модель определила: {predicted_class}')

    img_display = Image.open(image_path)
    plt.imshow(img_display)
    plt.title(f'Модель определила: {predicted_class}')
    plt.axis('off')
    plt.show()


predict_image('dataset/dog/000033.jpg')