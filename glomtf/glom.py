import tensorflow as tf
from einops.layers.tensorflow import Rearrange
from einops import rearrange, repeat
from .grouped_feed_forward import GroupedFeedForward
from .consensus_attention import ConsensusAttention


class Glom(tf.keras.Model):
    def __init__(
            self,
            dim=512,
            levels=6,
            image_size=224,
            patch_size=14,
            consensus_self=False,
            local_consensus_radius=0):

        super(Glom, self).__init__()
        num_patches_side = (image_size // patch_size)
        num_patches = num_patches_side ** 2
        self.levels = levels

        self.image_to_tokens = tf.keras.Sequential(
            [
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_size,
                    p2=patch_size),
                tf.keras.layers.Dense(
                    dim,
                    input_dim=patch_size ** 2 *
                    3)])
        self.pos_emb = tf.keras.layers.Embedding(num_patches, dim)
        self.init_levels = tf.Variable(tf.random.normal([levels, dim]))

        # bottom-up and top-down unit
        self.bottom_up = GroupedFeedForward(dim=dim,
                                            groups=levels)
        self.top_down = GroupedFeedForward(dim=dim,
                                           groups=levels - 1)

        # consensus attention unit
        self.attention = ConsensusAttention(
            num_patches_side,
            attend_self=consensus_self,
            local_consensus_radius=local_consensus_radius)

    def call(self, img, iters=None, levels=None, return_all=False):
        b = img.shape[0]
        h = img.shape[1]
        w = img.shape[2]
        # b, h, w, _ = *img.shape

        if iters is not None:
            iters = iters
        else:
            iters = self.levels * 2

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        pos_embs = self.pos_emb(tf.range(n))
        pos_embs = rearrange(pos_embs, 'n d -> () n () d')

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if levels is None:
            levels = repeat(self.init_levels, 'l d -> b n l d', b=b, n=n)

        hiddens = [levels]

        num_contributions = tf.Variable(tf.fill(self.levels, 4))
        num_contributions[-1].assign(3)

        for _ in range(iters):
            levels_with_input = tf.concat((bottom_level, levels), axis=-2)
            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])
            top_down_out = self.top_down(
                levels_with_input[..., 2:, :] + pos_embs)

            # [[0, 1],[0, 0]]
            top_down_out = tf.pad(
                top_down_out, [[0, 0], [0, 0], [0, 1], [0, 0]])
            consensus = self.attention(levels)

            levels_sum = tf.stack((levels,
                                   bottom_up_out,
                                   top_down_out,
                                   consensus))
            levels_sum = tf.reduce_sum(levels_sum, 0)

            rearranged_num_contributions = tf.cast(
                rearrange(num_contributions, 'l -> () () l ()'), tf.float32)

            levels_mean = levels_sum / rearranged_num_contributions

            levels = levels_mean
            hiddens.append(levels)

        if return_all:
            return tf.stack(hiddens)

        return levels
