import tensorflow as tf
from einops.layers.tensorflow import Rearrange


class GroupedFeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, groups, mult=4):
        super(GroupedFeedForward, self).__init__()
        total_dim = dim * groups
        self.net = tf.keras.Sequential(
            [
                Rearrange('b n l d -> b (l d) n'),
                tf.keras.layers.Conv1D(
                    total_dim,
                    total_dim * mult,
                    1,
                    groups=groups,
                    activation='gelu'),
                tf.keras.layers.Conv1D(
                    total_dim,
                    total_dim * mult,
                    1,
                    groups=groups),
                Rearrange(
                    'b (l d) n -> b n l d',
                    l=groups)])

    def call(self, inputs):
        return self.net(inputs)
