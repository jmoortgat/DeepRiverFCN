"""ArcticRiverMap: a fully convolutional encoder-decoder for river segmentation.

The network follows the DeepWaterMap-V2 architecture described by
Isikdogan, Bovik & Passalacqua, "Seeing Through the Clouds with
DeepWaterMap", IEEE Geoscience and Remote Sensing Letters, 2019
(doi:10.1109/LGRS.2019.2953261), adapted here for panchromatic (1-band)
and 4-band multispectral commercial satellite imagery as described in
Moortgat et al., Remote Sensing of Environment 282, 113279 (2022),
doi:10.1016/j.rse.2022.113279.

Structure: a 1x1 entry convolution, four stride-2 residual downscaling
stages (channel width x4 per stage), a residual bottleneck, and four
depth-to-space upscaling stages mirrored with additive skip connections,
closing with a 1x1 convolution and sigmoid to produce a single-channel
water probability map. All convolutions are bias-free and batch-normalized.
"""

import tensorflow as tf

__all__ = ["build_model", "model"]

_N_STAGES = 4  # encoder/decoder depth


def _conv_bn(x, filters, kernel_size, stride=1, relu=True):
    """Conv2D (no bias, he_uniform) -> BatchNorm -> optional ReLU."""
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if relu:
        x = tf.keras.layers.Activation("relu")(x)
    return x


def _residual_pair(x, filters, first_kernel=3, first_stride=1):
    """Two conv blocks whose outputs are summed (local residual unit)."""
    a = _conv_bn(x, filters, kernel_size=first_kernel, stride=first_stride)
    b = _conv_bn(a, filters, kernel_size=3)
    return tf.keras.layers.Add()([a, b])


def build_model(n_channels: int, min_width: int = 4) -> "tf.keras.Model":
    """Build the ArcticRiverMap network for ``n_channels`` input bands.

    Parameters
    ----------
    n_channels : int
        Number of image bands (1 for panchromatic, 4 for multispectral).
    min_width : int
        Channel width after the entry convolution; each encoder stage
        multiplies the width by 4.
    """
    inputs = tf.keras.layers.Input(shape=[None, None, n_channels])

    # Entry: project the bands to the base width (linear, no activation).
    x = _conv_bn(inputs, min_width, kernel_size=1, relu=False)

    # Encoder: stride-2 residual stages, quadrupling the width each time.
    skips = [x]
    for _ in range(_N_STAGES):
        width = 4 * int(x.shape[-1])
        x = _residual_pair(x, width, first_kernel=5, first_stride=2)
        skips.append(x)

    # Bottleneck at constant width.
    x = _residual_pair(x, int(x.shape[-1]))

    # Decoder: depth-to-space upsampling + additive skip connections.
    for _ in range(_N_STAGES):
        x = tf.keras.layers.Add()([x, skips.pop()])
        width = int(x.shape[-1]) // 4
        x = tf.keras.layers.Lambda(
            lambda t: tf.nn.depth_to_space(input=t, block_size=2)
        )(x)
        a = _conv_bn(x, width, kernel_size=3)
        b = _conv_bn(a, width, kernel_size=3)
        x = tf.keras.layers.Add()([a, b])

    # Exit: fold in the entry skip, project to one channel, sigmoid.
    x = tf.keras.layers.Add()([x, skips.pop()])
    x = _conv_bn(x, 1, kernel_size=1, relu=False)
    outputs = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def model(n_channels: int = 4, min_width: int = 4):
    """Backward-compatible alias for :func:`build_model`."""
    return build_model(n_channels=n_channels, min_width=min_width)
