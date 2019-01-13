# coding: utf-8
# https://github.com/broadinstitute/keras-resnet/blob/master/keras_resnet/models/_2d.py
# https://github.com/fizyr/keras-models/releases/tag/v0.0.1

from base_nets.resnet_base import *


def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    """
    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names, block=bottleneck_2d, include_top=include_top, classes=classes, *args, **kwargs)


def resnet50_model_v2(input_shape=(224, 224, 3), weights='imagenet'):
    import os
    from keras.layers import Input

    x = Input(input_shape)
    model = ResNet50(x, include_top=True)

    if weights == 'imagenet':
        print('Load imagenet weights...')
        weights_path = os.path.dirname(os.path.realpath(__file__)) + '/../../weights/ResNet-50-model.keras.h5'
        model.load_weights(weights_path)

    return model
