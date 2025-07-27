import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm import tqdm

# -------------------- CONFIG --------------------
IMG_SIZE = (128, 128)  # Resize all images to this size
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
TRAIN_DIR = r"D:\Downloads\cats_vs_dogs\Dataset\training_set"
TEST_DIR = r"D:\Downloads\cats_vs_dogs\Dataset\test_set"

# -------------------- HELPER FUNCTIONS --------------------
def extract_hog_features(image_path):
    try:
        image = imread(image_path)
        image = resize(image, IMG_SIZE, anti_aliasing=True)
        image = rgb2gray(image)
        features = hog(image, orientations=ORIENTATIONS, 
                       pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK,
                       block_norm='L2-Hys', feature_vector=True)
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_dataset(folder_path, label):
    features = []
    labels = []
    for file_name in tqdm(os.listdir(folder_path), desc=f"Loading {os.path.basename(folder_path)}"):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path):
            hog_feat = extract_hog_features(full_path)
            if hog_feat is not None:
                features.append(hog_feat)
                labels.append(label)
    return features, labels

# -------------------- LOAD TRAIN DATA --------------------
print("📦 Loading training data...")
cat_train, cat_labels = load_dataset(os.path.join(TRAIN_DIR, "cats"), 0)
dog_train, dog_labels = load_dataset(os.path.join(TRAIN_DIR, "dogs"), 1)

X = np.array(cat_train + dog_train)
y = np.array(cat_labels + dog_labels)

# -------------------- SPLIT AND TRAIN --------------------
print("🧠 Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print("🚀 Training SVM...")
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

# -------------------- EVALUATION --------------------
print("✅ Evaluating on validation set...")
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# -------------------- TESTING --------------------
print("🧪 Loading test data and predicting...")
cat_test, cat_test_labels = load_dataset(os.path.join(TEST_DIR, "cats"), 0)
dog_test, dog_test_labels = load_dataset(os.path.join(TEST_DIR, "dogs"), 1)

X_test = np.array(cat_test + dog_test)
y_test = np.array(cat_test_labels + dog_test_labels)

print("🧾 Predicting test set...")
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Report:\n", classification_report(y_test, y_test_pred))


