import sys
print(">>> Python đang chạy tại:", sys.executable)

import os
import cv2
cv2.startWindowThread()
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ============================================================
# 1. LOAD & GÁN NHÃN DỮ LIỆU
# ============================================================
def load_dataset(dataset_path, img_size=(64, 64)):
    X = []
    y = []
    label_map = {}

    folders = os.listdir(dataset_path)

    for idx, folder in enumerate(folders):
        label_map[idx] = folder
        print(f"> Đang load lớp: {folder}")

        folder_path = os.path.join(dataset_path, folder)

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, img_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            X.append(gray)
            y.append(idx)

    print("\n> Hoàn tất load dữ liệu!")
    return np.array(X), np.array(y), label_map



# ============================================================
# 2. TRÍCH ĐẶC TRƯNG HOG
# ============================================================
def extract_hog_features(images):
    features = []
    print("\n> Đang trích HOG...")

    for img in images:
        hog_vector = hog(img,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         block_norm='L2-Hys')

        features.append(hog_vector)

    print("> Hoàn tất trích HOG!\n")
    return np.array(features)



# ============================================================
# 3. HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH SVM
# ============================================================
def train_model(dataset_path="dataset"):
    # Load data
    X, y, label_map = load_dataset(dataset_path)

    # Extract HOG
    X_hog = extract_hog_features(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y, test_size=0.2, random_state=42
    )

    # Train SVM
    print("> Đang huấn luyện mô hình SVM...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=======================")
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=======================")
    print("Accuracy:", acc)
    from sklearn.utils.multiclass import unique_labels

    labels = unique_labels(y_test, y_pred)
    target_names = [list(label_map.values())[i] for i in labels]

    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

    # Save model
    joblib.dump(model, "traffic_sign_svm.pkl")
    joblib.dump(label_map, "label_map.pkl")
    print("\n> Đã lưu mô hình: traffic_sign_svm.pkl")
    print("> Đã lưu nhãn: label_map.pkl")



# ============================================================
# 4. NHẬN ẢNH TỪ NV1 ĐỂ PHÂN LOẠI + HIỂN THỊ ẢNH
# ============================================================
def classify_from_nv1(img_path):
    cv2.namedWindow("Kết quả phân loại", cv2.WINDOW_NORMAL)

    print(f"\n> Phân loại ảnh NV1: {img_path}")

    model = joblib.load("traffic_sign_svm.pkl")
    label_map = joblib.load("label_map.pkl")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Không đọc được ảnh!")
        return None

    # Resize for model
    resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Extract HOG for this image
    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')

    # Predict
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0].max()

    label_text = f"{label_map[pred]} ({prob*100:.1f}%)"
    print(f"> Kết quả: {label_map[pred]}")
    print(f"> Độ tin cậy: {prob*100:.2f}%\n")

    # ============================
    # HIỂN THỊ ẢNH + NHÃN DỰ ĐOÁN
    # ============================
    img_show = img.copy()
    cv2.putText(img_show, label_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Kết quả phân loại", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return label_map[pred]


# ============================================================
# 5. MAIN MENU – CHO PHÉP CHẠY TRAIN HOẶC TEST
# ============================================================
if __name__ == "__main__":
    print("===============================")
    print(" PHÂN LOẠI BIỂN BÁO GIAO THÔNG ")
    print("===============================")
    print("1. Huấn luyện mô hình")
    print("2. Phân loại ảnh từ NV1")
    print("3. Thoát")

    choice = input("\nChọn chức năng (1/2/3): ")

    if choice == "1":
        train_model()

    elif choice == "2":
        img_path = input("Nhập đường dẫn ảnh ROI NV1: ")
        classify_from_nv1(img_path)

    else:
        print("Thoát chương trình.")
