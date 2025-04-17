import cv2
import numpy as np
import pickle  # Using pickle instead of joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (replace with actual image paths)
leaf_images = [...]  # List of paths to leaf images
non_leaf_images = [...]  # List of paths to non-leaf images
labels = [1] * len(leaf_images) + [0] * len(non_leaf_images)  # 1 = Leaf, 0 = Non-Leaf

# Feature Extraction using HOG
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize for uniformity
    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    return features

# Extract features
features = [extract_features(img) for img in (leaf_images + non_leaf_images)]

# Train SVM
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)

# Save the trained model using pickle

pickle.dump(svm_model, f)

# Evaluate accuracy
y_pred = loaded_model.predict(X_test)
print("Leaf vs. Non-Leaf Classification Accuracy:", accuracy_score(y_test, y_pred))
