# preprocess.py

import cv2
import numpy as np

# === Step 3: Character Mapping ===
CHARS = '0123456789'
char_to_num = {c: i for i, c in enumerate(CHARS)}
num_to_char = {i: c for c, i in char_to_num.items()}

def preprocess_image(img_path, width=128, height=32):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (width, height))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img
