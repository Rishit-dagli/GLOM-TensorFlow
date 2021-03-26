import tensorflow as tf


def pairwise_dist(A, B):
    """Write an algorithm that computes batched the p-norm distance between each pair of two collections of row vectors.

    We use the euclidean distance metric.
    For a matrix A [m, d] and a matrix B [n, d] we expect a matrix of
    pairwise distances here D [m, n]

    # Arguments:
        A: A tf.Tensor object. The first matrix.
        B: A tf.tensor object. The second matrix.

    # Returns:
        Calculate distance.

    # Reference:
        [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
        [tensorflow/tensorflow#30659](https://github.com/tensorflow/tensorflow/issues/30659)
    """

    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidean difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D
