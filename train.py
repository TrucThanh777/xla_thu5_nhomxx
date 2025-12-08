import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ============================================================
# 1. KIỂM TRA DATASET
# ============================================================
def check_dataset(path):
    if not os.path.exists(path):
        raise Exception(f"❌ Không tìm thấy thư mục dataset: {path}")

    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(classes) < 2:
        raise Exception("❌ Dataset phải có ÍT NHẤT 2 lớp (VD: Cam, NguyHiem, ChiDan…)")

    print("✔ Dataset OK! Các lớp tìm thấy:", classes)
    return classes


# ============================================================
# 2. LOAD DATASET
# ============================================================
def load_dataset(dataset_path, img_size=(64,64)):
    X, y = [], []
    label_map = {}

    classes = check_dataset(dataset_path)

    for idx, folder in enumerate(classes):
        label_map[idx] = folder
        folder_path = os.path.join(dataset_path, folder)

        print(f"> Load lớp [{folder}] ...")

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                print("⚠ Bỏ qua file lỗi:", img_path)
                continue

            img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X.append(gray)
            y.append(idx)

    print(f"✔ Load xong: {len(X)} ảnh")
    return np.array(X), np.array(y), label_map


# ============================================================
# 3. TRÍCH ĐẶC TRƯNG HOG
# ============================================================
def extract_hog_features(images):
    print("> Trích đặc trưng HOG ...")

    features = []
    for img in images:
        hog_vec = hog(img,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys')

        features.append(hog_vec)

    print("✔ HOG hoàn tất!")
    return np.array(features)


# ============================================================
# 4. TRAIN + SAVE MODEL
# ============================================================
def train_model():
    DATASET = "dataset"

    print("\n===== BẮT ĐẦU TRAIN MODEL =====")

    X, y, label_map = load_dataset(DATASET)
    X_hog = extract_hog_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y, test_size=0.2, random_state=42
    )

    print("> Đang huấn luyện SVM ...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print("Accuracy:", acc)

    # Không crash target_name nữa
    print(classification_report(y_test, y_pred, zero_division=0))

    # Lưu file
    joblib.dump(model, "traffic_sign_svm.pkl")
    joblib.dump(label_map, "label_map.pkl")

    print("\n🎉 TRAIN THÀNH CÔNG!")
    print("✔ Đã lưu model: traffic_sign_svm.pkl")
    print("✔ Đã lưu nhãn: label_map.pkl\n")


if __name__ == "__main__":
    train_model()
