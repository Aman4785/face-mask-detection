import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide main Tkinter window
Tk().withdraw()

# Open file dialog
file_path = askopenfilename(title="Select an Image")

if file_path == "":
    print("No file selected!")
    exit()

# Load model
model = load_model("mask_detector_model.h5")

# Load image
img = cv2.imread(file_path)

if img is None:
    print("Error loading image!")
    exit()

# Preprocess
img_resized = cv2.resize(img, (128,128))
img_norm = img_resized / 255.0
img_input = np.reshape(img_norm, (1,128,128,3))

# Prediction
prediction = model.predict(img_input)
label = "MASK" if np.argmax(prediction)==0 else "NO MASK"

print("Prediction:", label)

# Show result
cv2.putText(img, label, (30,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()