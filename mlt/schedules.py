import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, initial_steps=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.initial_steps = initial_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_steps)
        arg2 = (step + self.initial_steps) * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return dict([("d_model", self.d_model.numpy()),
                     ("initial_steps", self.initial_steps),
                     ("warmup_steps", self.warmup_steps)])
