import tensorflow as tf
from tensorflow.keras import backend as K


class Model:
    def __init__(self, model_path):
        self._model_path = model_path
        self._model = self._load_model()

    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def dice_coefficient(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def combined_loss(y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + Model.dice_loss(y_true, y_pred)

    def _load_model(self):
        model = tf.keras.models.load_model(self._model_path, custom_objects={
            'dice_coefficient': self.dice_coefficient,
            'dice_loss': self.dice_loss,
            'combined_loss': self.combined_loss
        })
        return model

    def get_model(self):
        return self._model
