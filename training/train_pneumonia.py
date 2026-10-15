import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH = 16
EPOCHS = 12

data_path = r"C:\Users\USER\Desktop\AI_Disease_Prediction_System(new)\dataset\pneumonia"

# Data generator with automatic train/validation split
gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train = gen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

val = gen.flow_from_directory(
    data_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)
print("Classes:", train.class_indices)
print("Train samples:", train.samples)
print("Validation samples:", val.samples)

# Load ResNet50
base = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze most layers (faster CPU training)
for layer in base.layers[:-20]:
    layer.trainable = False

# Custom classifier
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(train.num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=base.input, outputs=output)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
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

# Train
model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# Save model
model.save("../models/pneumonia_model.h5")

print("✅ Pneumonia Model Saved Successfully")