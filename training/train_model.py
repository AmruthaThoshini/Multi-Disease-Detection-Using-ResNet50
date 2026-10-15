import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Image settings
IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 10

# Dataset location
DATASET_PATH = "C:/Users/USER/Desktop/AI_Disease_Prediction_System(new)/dataset"

# Model save location
MODEL_PATH = "C:/Users/USER/Desktop/AI_Disease_Prediction_System(new)/models"

# Create models folder if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training data
train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset="training"
)

# Validation data
val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset="validation"
)

# Load ResNet50 base model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS
)

# Save trained model
model.save(MODEL_PATH + "/resnet_model.h5")

print("✅ Model saved successfully!")