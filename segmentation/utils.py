import numpy as np
import cv2

from model.model import Model

model = Model('C:/\/Users\MEDIA\Desktop\LiveCELL\model\livecell_full_segmentation1.h5')
model = model.get_model()


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    processed_image = cv2.resize(image, (256, 256))  # Example resize
    processed_image = processed_image / 255.0  # Example normalization
    return processed_image


def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    # Post-process predictions to extract instance segmentation results
    return predictions
