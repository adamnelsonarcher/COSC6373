import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras import ops

import kagglehub

download_root = Path(kagglehub.dataset_download("akrsnv/horses-and-camels"))

image_size = 224
all_images = []
all_labels = []
class_names = ["camel", "horse"]

for split_folder in ["train", "test"]:
    for label_idx, cls_name in enumerate(class_names):
        folder = download_root / split_folder / cls_name
        if not folder.exists():
            continue
        for img_path in sorted(folder.glob("*.png")):
            img = load_img(img_path, target_size=(image_size, image_size))
            all_images.append(img_to_array(img))
            all_labels.append(label_idx)

images = np.array(all_images, dtype="float32") / 255.0
labels_arr = np.array(all_labels)

x_temp, x_test, y_temp, y_test = train_test_split(
    images, labels_arr, test_size=0.15, random_state=42, stratify=labels_arr
)

# Rebuild CNN
def create_cnn_classifier():
    model = keras.Sequential([
        keras.Input(shape=(image_size, image_size, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ])
    return model

cnn = create_cnn_classifier()
cnn.load_weights("cnn_checkpoint.weights.h5")
cnn_preds = np.argmax(cnn.predict(x_test, verbose=0), axis=1)
print("CNN CM:")
print(confusion_matrix(y_test, cnn_preds))

