import cv2
import numpy as np
from config import IMAGE_SIZE

def preprocess_image(path):

    img = cv2.imread(path)

    img = cv2.resize(img, IMAGE_SIZE)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img