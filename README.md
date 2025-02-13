import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = "path_to_kaggle_dataset"

image_size = (128, 128)
data = []
labels = []

for category in ["Cat", "Dog"]:
    folder_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append(img)
            labels.append(category)

data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Cat -> 0, Dog -> 1

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  

def extract_features(images):
    features = base_model.predict(images)
    return features.reshape(features.shape[0], -1)

features = extract_features(data)

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Classification Accuracy: {accuracy:.4f}")
