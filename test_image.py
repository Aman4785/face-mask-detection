import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide Tkinter window
root = Tk()
root.withdraw()

# Select image
file_path = askopenfilename(
    title="Select Group Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if file_path == "":
    print("No file selected!")
    exit()

# Load model
model = load_model("mask_detector_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread(file_path)

if img is None:
    print("Error loading image!")
    exit()

# 🔥 Resize large image BEFORE detection (better performance)
max_width = 800
h, w = img.shape[:2]

if w > max_width:
    scale = max_width / w
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 🔥 Improve contrast
gray = cv2.equalizeHist(gray)

# 🔥 Improved face detection
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(40, 40)
)

if len(faces) == 0:
    print("No faces detected!")

used_positions = []
mask_count = 0
no_mask_count = 0

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]

    # Preprocess for model
    face_resized = cv2.resize(face, (128,128))
    face_norm = face_resized / 255.0
    face_input = np.reshape(face_norm, (1,128,128,3))

    # Prediction
    prediction = model.predict(face_input)
    confidence = np.max(prediction) * 100

    if np.argmax(prediction) == 0:
        label = f"MASK ({confidence:.1f}%)"
        color = (0,255,0)
        mask_count += 1
    else:
        label = f"NO MASK ({confidence:.1f}%)"
        color = (0,0,255)
        no_mask_count += 1

    # Draw bounding box
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)

    # Decide label position
    text_y = y - 10 if y - 10 > 10 else y + h + 25

    # Avoid overlap
    for prev_x, prev_y in used_positions:
        if abs(text_y - prev_y) < 20 and abs(x - prev_x) < 100:
            text_y += 25

    used_positions.append((x, text_y))

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    # Draw background
    cv2.rectangle(
        img,
        (x, text_y - text_height - 5),
        (x + text_width, text_y + 5),
        color,
        -1
    )

    # Put label text
    cv2.putText(
        img,
        label,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

# 🔥 Summary display
summary = f"Mask: {mask_count}   No Mask: {no_mask_count}"
cv2.rectangle(img, (10,10), (350,50), (0,0,0), -1)
cv2.putText(img, summary, (15,40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# 🔥 FINAL SCREEN-FIT SCALING (no overflow)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

max_width = int(screen_width * 0.9)
max_height = int(screen_height * 0.9)

h, w = img.shape[:2]

scale_w = max_width / w
scale_h = max_height / h

scale = min(scale_w, scale_h, 1.0)  # never upscale

new_width = int(w * scale)
new_height = int(h * scale)

img = cv2.resize(img, (new_width, new_height))

# 🔥 Resizable window
cv2.namedWindow("Group Mask Detection", cv2.WINDOW_NORMAL)

# Show result
cv2.imshow("Group Mask Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()