import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import os

# 🔥 Load image correctly
def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".webp", ".avif"]:
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert ONLY here
        return img
    else:
        return cv2.imread(path)

# GUI
root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="Select Image",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.avif")]
)

if file_path == "":
    print("No file selected!")
    exit()

# Load model
model = load_model("mask_detector_model.h5")

# Load detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load image
img = load_image(file_path)

if img is None:
    print("Error loading image!")
    exit()

# Resize large images
max_width = 800
h, w = img.shape[:2]

if w > max_width:
    scale = max_width / w
    img = cv2.resize(img, (int(w*scale), int(h*scale)))

# Grayscale + improve contrast
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(40, 40)
)

# 🔥 FIX: fallback if no face
if len(faces) == 0:
    faces = [(0, 0, img.shape[1], img.shape[0])]

used_positions = []
mask_count = 0
no_mask_count = 0

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]

    face_resized = cv2.resize(face, (128,128))
    face_norm = face_resized / 255.0
    face_input = np.reshape(face_norm, (1,128,128,3))

    prediction = model.predict(face_input, verbose=0)
    confidence = np.max(prediction) * 100

    if np.argmax(prediction) == 0:
        label = f"MASK ({confidence:.1f}%)"
        color = (0,255,0)
        mask_count += 1
    else:
        label = f"NO MASK ({confidence:.1f}%)"
        color = (0,0,255)
        no_mask_count += 1

    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

    text_y = y - 10 if y - 10 > 10 else y + h + 25

    for px, py in used_positions:
        if abs(text_y - py) < 20 and abs(x - px) < 100:
            text_y += 25

    used_positions.append((x, text_y))

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(img, (x, text_y - th - 5), (x + tw, text_y + 5), color, -1)

    cv2.putText(img, label, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255,255,255), 2)

# Summary
summary = f"Mask: {mask_count}   No Mask: {no_mask_count}"
cv2.rectangle(img, (10,10), (350,50), (0,0,0), -1)
cv2.putText(img, summary, (15,40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# Fit to screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

max_w = int(screen_width * 0.9)
max_h = int(screen_height * 0.9)

h, w = img.shape[:2]
scale = min(max_w / w, max_h / h, 1.0)

img = cv2.resize(img, (int(w*scale), int(h*scale)))

# Show
cv2.namedWindow("Smart Mask Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Smart Mask Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()