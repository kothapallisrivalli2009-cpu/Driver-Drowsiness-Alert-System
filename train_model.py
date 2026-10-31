
import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset folder path
DATASET_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = (24, 24)

X = []
y = []

# 2. Load images from both folders
folders = {
    "Closed Eyes": 0,
    "Open Eyes": 1
}

for folder_name, label in folders.items():

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.isdir(folder_path):
        print("Folder not found:", folder_path)
        raise SystemExit

    count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:

            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue

                image = cv2.resize(image, IMAGE_SIZE)
                image = image.astype(np.float32) / 255.0

                X.append(image.flatten())
                y.append(label)
                count += 1

    print(f"{folder_name}: {count} images loaded")

# 3. Check dataset
if len(X) == 0:
    print("No images loaded. Check your folder structure.")
    raise SystemExit

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nTotal images:", len(X))
print("Closed Eyes:", np.sum(y == 0))
print("Open Eyes:", np.sum(y == 1))

if len(np.unique(y)) < 2:
    print("Error: Both Open Eyes and Closed Eyes images are required.")
    raise SystemExit

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining the ML model...")

# 5. Create and train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Closed Eyes", "Open Eyes"],
        zero_division=0
    )
)

# 7. Save the trained model
model_path = os.path.join(DATASET_PATH, "drowsiness_model.pkl")

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)
