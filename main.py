import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load Data
data = []
labels = []

categories = ["with_mask", "without_mask"]
path = "dataset"

for category in categories:
    folder_path = os.path.join(path, category)
    label = categories.index(category)

    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)

        try:
            image = cv2.resize(image, (128, 128))
            data.append(image)
            labels.append(label)
        except:
            pass

# Preprocess
data = np.array(data) / 255.0
labels = np.array(labels)
labels = to_categorical(labels, 2)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)


# Build Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Save Model
model.save("mask_detector_model.h5")