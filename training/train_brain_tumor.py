import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH = 16
EPOCHS = 10

data_path = r"C:\Users\USER\Desktop\AI_Disease_Prediction_System(new)\dataset\brain_tumor\Training"

# Data generator with augmentation
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

# Load ResNet50
base = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze early layers (train last layers)
for layer in base.layers[:-30]:
    layer.trainable = False

# Custom classifier
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(train.num_classes, activation="softmax")(x)

model = tf.keras.Model(base.input, output)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    steps_per_epoch=100
)

# Save model
model.save("../models/brain_tumor_model.h5")

print("✅ Brain Tumor Model Saved Successfully")