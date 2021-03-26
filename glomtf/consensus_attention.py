import tensorflow as tf
from einops import rearrange
from .pairwisedist import pairwise_dist


class ConsensusAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            num_patches_side,
            attend_self=True,
            local_consensus_radius=0):
        super(ConsensusAttention, self).__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coors = tf.cast(tf.stack(tf.meshgrid(
                tf.range(num_patches_side),
                tf.range(num_patches_side),
                indexing='ij'
            )), "float")

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = pairwise_dist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')

            self.non_local_mask = mask_non_local
            self.TOKEN_ATTEND_SELF_VALUE = -5e-4

    def call(self, levels):
        n = levels.shape[1]
        d = levels.shape[3]
        # _, n, _, d = levels.shape

        q, v = levels, levels
        k = tf.math.l2_normalize(levels)

        sim = tf.einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = tf.eye(n, dtype='bool')
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim = tf.where(self_mask, -5e-4, sim)

        if self.local_consensus_radius > 0:
            max_neg_value = -tf.experimental.numpy.finfo(sim.stype).max
            sim = tf.where(self.non_local_mask, max_neg_value, sim)

        attn = tf.nn.softmax(sim, axis=-1)
        out = tf.einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out
