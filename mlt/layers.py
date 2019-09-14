import math

import tensorflow as tf
from tensorflow.keras import constraints, initializers, regularizers  # pylint: disable=import-error
from tensorflow.keras.layers import Layer  # pylint: disable=import-error


class Attention(Layer):
    def call(self, inputs):
        q, k, v, mask = inputs

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights


class ConditionalNormalization(Layer):
    """
    https://github.com/suyash/stylizer/blob/master/stylizer/layers/normalization.py#L6
    """
    def __init__(
            self,
            num_factors,
            axis,  # the axis along which the inputs are normalized
            epsilon=1e-3,
            beta_initializer="zeros",
            gamma_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs):
        super().__init__(**kwargs)

        # normalize either along the time steps, or the channels
        # axis == -1 is layer norm
        # axis == 1 is instance norm
        assert axis == 1 or axis == -1

        self.num_factors = num_factors
        self.axis = axis
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=[self.num_factors, input_shape[0][-1]],
            name="gamma",
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint)

        self.beta = self.add_weight(
            shape=[self.num_factors, input_shape[0][-1]],
            name="beta",
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint)

    def call(self, inputs):
        """
        inputs:
            x: [batch_size, time_steps, d_size]
            factors: [batch_size, num_factors]

        first normalizes the unnormalized input,
        and then multiplies it with a weighted sum of the gamma and beta parameters
        """

        x, factors = inputs

        mu, sigma_sq = tf.nn.moments(x, axes=self.axis, keepdims=True)
        normalized = (x - mu) / tf.sqrt(sigma_sq + self.epsilon)

        gamma = tf.expand_dims(factors, -1) * self.gamma
        beta = tf.expand_dims(factors, -1) * self.beta

        gamma = tf.reduce_sum(gamma, axis=1, keepdims=True)
        beta = tf.reduce_sum(beta, axis=1, keepdims=True)

        return gamma * normalized + beta

    def get_config(self):
        base = super().get_config()
        return dict(
            list(base.items()) + [
                ("num_factors", self.num_factors),
                ("axis", self.axis),
                ("epsilon", self.epsilon),
                ("beta_initializer",
                 initializers.serialize(self.beta_initializer)),
                ("gamma_initializer",
                 initializers.serialize(self.gamma_initializer)),
                ("beta_regularizer",
                 regularizers.serialize(self.beta_regularizer)),
                ("gamma_regularizer",
                 regularizers.serialize(self.gamma_regularizer)),
                ("beta_constraint",
                 constraints.serialize(self.beta_constraint)),
                ("gamma_constraint",
                 constraints.serialize(self.gamma_constraint)),
            ])


class Gelu(Layer):
    """
    - https://arxiv.org/abs/1606.08415
    - https://github.com/google-research/bert/blob/master/modeling.py#L264-L277
    - https://github.com/hendrycks/GELUs
    """
    def __init__(self, faster_approx=False, **kwargs):
        super().__init__(**kwargs)
        self.faster_approx = faster_approx

    def call(self, x):
        if self.faster_approx:
            cdf = tf.sigmoid(1.072 * x)
        else:
            cdf = 0.5 * (1.0 + tf.tanh(
                (tf.math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))

        return x * cdf

    def get_config(self):
        base = super().get_config()
        return dict(
            list(base.items()) + [("faster_approx", self.faster_approx)])


class MultiplyConstant(Layer):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def call(self, inputs):
        return inputs * self.c

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("c", self.c)])


class PaddingMask(Layer):
    def call(self, inputs):
        seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]


class PaddingAndLookaheadMask(Layer):
    def call(self, inputs):
        size = tf.shape(inputs)[1]
        lhm = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        seq = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        seq = seq[:, tf.newaxis, tf.newaxis, :]

        return tf.maximum(lhm, seq)


class PositionalEncoding(Layer):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model

    def call(self, inputs):
        position = tf.shape(inputs)[1]

        position_dims = tf.range(position)[:, tf.newaxis]
        embed_dims = tf.range(self.d_model)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(
            10000.0, tf.cast(
                (2 * (embed_dims // 2)) / self.d_model, tf.float32))
        angle_rads = tf.cast(position_dims, tf.float32) * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_config(self):
        base = super().get_config()
        return dict(list(base.items()) + [("d_model", self.d_model)])
