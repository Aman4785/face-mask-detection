import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))

    # Prediction
    prediction = model.predict(img)
    label = "MASK" if np.argmax(prediction)==0 else "NO MASK"

    # Display text
    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()