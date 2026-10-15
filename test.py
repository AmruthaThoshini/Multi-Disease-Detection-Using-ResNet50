from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# load model
model = load_model("models/retinopathy_model.h5")

# load test image
img = image.load_img("test.jpg", target_size=(224,224))
img = image.img_to_array(img)
img = img/255.0
img = np.expand_dims(img, axis=0)

# predict
prediction = model.predict(img)

print("Prediction:", prediction)