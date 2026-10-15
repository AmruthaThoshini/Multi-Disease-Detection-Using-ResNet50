import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH = 16
EPOCHS = 12

# ==============================
# DATASET PATH
# ==============================

train_path = r"C:\Users\USER\Desktop\AI_Disease_Prediction_System(new)\dataset\diabetic_retinopathy\Dataset\train"

# ==============================
# DATA GENERATOR
# ==============================

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

print("Detected Classes:", train_data.class_indices)

# ==============================
# MODEL (TRANSFER LEARNING)
# ==============================

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# CALLBACKS
# ==============================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2
)

# ==============================
# TRAIN MODEL
# ==============================

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ==============================
# SAVE MODEL
# ==============================

os.makedirs("../models",exist_ok=True)

model.save("../models/eye_disease_model.h5")

print("✅ Eye Disease Model Saved Successfully")