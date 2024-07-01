import numpy as np
from PIL import Image


def process_prediction(prediction_list):
    prediction_array = np.array(prediction_list, dtype=np.uint8)
    prediction_array = np.squeeze(prediction_array, axis=2)
    prediction_image = Image.fromarray(prediction_array, mode='L')
    return prediction_image


def resize_image(image_path, size=(256, 256)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image
