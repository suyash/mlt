import tensorflow as tf


class SparseSoftmaxCrossentropyWithMaskedPadding(tf.keras.losses.Loss):
    def __init__(self, mask_val, **kwargs):
        super().__init__(**kwargs)
        self.mask_val = mask_val

    def call(self, y_true, y_pred):
        loss = tf.keras.backend.sparse_categorical_crossentropy(
            target=y_true, output=y_pred, from_logits=True, axis=-1)

        mask = tf.not_equal(y_true, self.mask_val)
        mask = tf.reshape(mask, tf.shape(loss))
        mask = tf.cast(mask, tf.float32)

        return loss * mask

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("mask_val", self.mask_val)])
