import numpy as np

def predict(model, img):

    predictions = model.predict(img)[0]

    print("Prediction probabilities:", predictions)

    class_index = int(np.argmax(predictions))

    confidence = float(predictions[class_index]) * 100

    return class_index, round(confidence, 2)