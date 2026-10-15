import os
import uuid
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, HEATMAP_FOLDER
from utils.preprocessing import preprocess_image
from utils.predict import predict
from utils.gradcam import generate_gradcam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ====================================
# LOAD MODELS
# ====================================

print("Loading Models...")

brain_model = tf.keras.models.load_model("models/brain_tumor_model.h5")
pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")
eye_model = tf.keras.models.load_model("models/eye_disease_model.h5")

print("Models Loaded Successfully")


# ====================================
# CLASS LABELS
# ====================================

brain_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

pneumonia_labels = ["Normal", "Pneumonia"]

eye_labels = ["AMD", "Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]


# ====================================
# BRAIN TUMOR INFO
# ====================================

tumor_info = {

"Glioma":
"Glioma is a tumor that occurs in the brain and spinal cord. It develops from glial cells and can affect brain function depending on its location.",

"Meningioma":
"Meningioma is a tumor that arises from the meninges surrounding the brain and spinal cord. It is usually slow-growing.",

"Pituitary":
"Pituitary tumors occur in the pituitary gland and may affect hormone production.",

"No Tumor":
"No tumor detected. The MRI scan appears normal."
}


# ====================================
# PNEUMONIA INFO
# ====================================

pneumonia_info = {

"Normal":
"The chest X-ray appears normal with clear lungs and no visible infection.",

"Pneumonia":
"Pneumonia is a lung infection that causes inflammation in the air sacs."
}


# ====================================
# EYE DISEASE INFO
# ====================================

eye_info = {

"AMD":
"Age-related Macular Degeneration affects the central retina and can reduce central vision.",

"Cataract":
"Cataract occurs when the eye lens becomes cloudy, causing blurred vision.",

"Glaucoma":
"Glaucoma damages the optic nerve due to increased eye pressure.",

"Diabetic Retinopathy":
"Diabetic Retinopathy damages the blood vessels of the retina due to diabetes.",

"Normal":
"The retina appears healthy with no visible signs of disease."
}


# ====================================
# HOME PAGE
# ====================================

@app.route('/')
def home():
    return render_template("index.html")


# ====================================
# PREDICTION
# ====================================

@app.route('/predict', methods=['POST'])
def predict_disease():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    disease_type = request.form['disease']

    if file.filename == '':
        return "No selected file"

    # SAVE IMAGE
    filename = secure_filename(file.filename)
    unique_name = str(uuid.uuid4()) + "_" + filename

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    # PREPROCESS IMAGE
    img = preprocess_image(filepath)

    # SELECT MODEL
    if disease_type == "brain":
        model = brain_model
    elif disease_type == "pneumonia":
        model = pneumonia_model
    elif disease_type == "eye":
        model = eye_model
    else:
        return "Invalid Disease Type"

    # PREDICTION
    class_index, confidence = predict(model, img)

    # MAP LABEL
    if disease_type == "brain":
        label = brain_labels[class_index]

    elif disease_type == "pneumonia":
        label = pneumonia_labels[class_index]

    else:
        label = eye_labels[class_index]

    # ====================================
    # RISK LEVEL (DISEASE SPECIFIC)
    # ====================================

    if label in ["Normal", "No Tumor"]:
        risk_level = "No Risk"

        if disease_type == "brain":
            advice = "No tumor detected. The MRI scan appears normal."

        elif disease_type == "pneumonia":
            advice = "No pneumonia detected. The chest X-ray appears normal."

        else:
            advice = "No eye disease detected. The retina appears healthy."

    else:

        if confidence <= 40:
            risk_level = "Low Risk"
            advice = "Early indicators detected. Monitoring recommended."

        elif confidence <= 70:
            risk_level = "Moderate Risk"
            advice = "Possible disease signs detected. Medical consultation recommended."

        else:
            risk_level = "High Risk"
            advice = "Strong disease indicators detected. Immediate medical consultation advised."

    # ====================================
    # DESCRIPTION
    # ====================================

    if disease_type == "brain":
        description = tumor_info.get(label, "")

    elif disease_type == "pneumonia":
        description = pneumonia_info.get(label, "")

    else:
        description = eye_info.get(label, "")

    # ====================================
    # GRADCAM
    # ====================================

    heatmap_path = generate_gradcam(model, filepath, class_index)

    # ====================================
    # RESULT
    # ====================================

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence,
        risk_level=risk_level,
        advice=advice,
        description=description,
        image_path="uploads/" + unique_name,
        heatmap_path=heatmap_path
    )


# ====================================
# RUN SERVER
# ====================================

if __name__ == "__main__":

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)

    app.run(debug=True)