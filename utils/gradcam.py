import tensorflow as tf
import cv2
import numpy as np
import os
import uuid
from config import HEATMAP_FOLDER


def generate_gradcam(model, img_path, class_index):

    # Read image
    img = cv2.imread(img_path)

    # Resize image for model
    img_resized = cv2.resize(img, (224, 224))

    # Normalize
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Last convolution layer (ResNet50)
    last_conv_layer = model.get_layer("conv5_block3_out")

    # Create GradCAM model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Create heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Generate unique filename
    filename = "gradcam_" + str(uuid.uuid4()) + ".jpg"

    save_path = os.path.join(HEATMAP_FOLDER, filename)

    # Save image
    cv2.imwrite(save_path, superimposed)

    # Return relative path for HTML
    return "heatmaps/" + filename