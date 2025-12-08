import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.feature import hog
import joblib

# ===================== LOAD MODEL =====================
model = joblib.load("traffic_sign_svm.pkl")
label_map = joblib.load("label_map.pkl")

# ===================== PHÂN LOẠI ======================
def classify_image(img_path):
    img = cv2.imread(img_path)

    resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    features = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')

    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0].max()

    return img, label_map[pred], prob


# ===================== MỞ FILE ======================
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return
    
    img, label, prob = classify_image(file_path)

    # Hiển thị ảnh lên GUI
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    img_label.config(image=img_tk)
    img_label.image = img_tk

    result_label.config(
        text=f"Kết quả: {label}\nĐộ tin cậy: {prob*100:.2f}%"
    )


# ===================== GIAO DIỆN CHÍNH ======================
root = Tk()
root.title("PHÂN LOẠI BIỂN BÁO GIAO THÔNG")
root.geometry("600x700")

title = Label(root, text="PHÂN LOẠI BIỂN BÁO", font=("Arial", 22, "bold"))
title.pack(pady=10)

btn = Button(root, text="Chọn ảnh", font=("Arial", 16), command=open_file)
btn.pack(pady=10)

img_label = Label(root)
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=20)

root.mainloop()
