# train_model.py

import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------- PATHS ----------------
train_dir = r"C:\Users\pc\Desktop\project_7_sem\AI_Smart_Skin_Care\Oily-Dry-Skin-Types\train"
val_dir = r"C:\Users\pc\Desktop\project_7_sem\AI_Smart_Skin_Care\Oily-Dry-Skin-Types\valid"

# ---------------- DATA GENERATOR ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# ---------------- SAVE CLASS INDICES ----------------
class_indices = train_gen.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)
print("Class Indices Saved:", class_indices)

# ---------------- CNN MODEL ----------------
num_classes = len(class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')  # Dynamic number of classes
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# ---------------- SAVE MODEL ----------------
model.save("skin_type_model.h5")
print("✅ Model saved successfully as skin_type_model.h5")
