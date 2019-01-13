# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


import numpy as np
import cv2
import random
import pandas as pd


def add_gaussian_noise(img, param=0.1):
    gaussian1 = np.random.random((img.shape[0], img.shape[1], 1)).astype(np.float32)
    gaussian2 = np.random.random((img.shape[0], img.shape[1], 1)).astype(np.float32)
    gaussian3 = np.random.random((img.shape[0], img.shape[1], 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian1, gaussian2, gaussian3), axis=2)
    r = random.uniform(0, param)

    gaussian_img = cv2.addWeighted(img.astype(np.uint8), 1 - r, (img.astype(np.float32) * gaussian).astype(np.uint8), r, 0)
    return gaussian_img


def apply_brightness_renormalization(img, clipLimitMin=3, clipLimitMax=3, tileGridSizeMin=8, tileGridSizeMax=10):
    # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    cl = random.randint(clipLimitMin, clipLimitMax)
    tl = random.randint(tileGridSizeMin, tileGridSizeMax)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(tl, tl))
    out = np.zeros_like(img)
    out[:, :, 0] = clahe.apply(img[:, :, 0])
    out[:, :, 1] = clahe.apply(img[:, :, 1])
    out[:, :, 2] = clahe.apply(img[:, :, 2])
    return out


def random_brightness_change(img, start_change=0, end_change=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert it to hsv
    value = random.randint(start_change, end_change)
    # for i in range(3):
    #    print(hsv[:, :, i].min(), hsv[:, :, i].max())
    hsv = hsv.astype(np.int16)
    hsv[:, :, 2] += value
    hsv[:, :, 2][hsv[:, :, 2] < 0] = 0
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def random_dynamic_range_change(img, start_min_change=0, start_max_change=20, end_min_change=0, end_max_change=20, separate_channel=False):
    img = img.astype(np.float32)

    if separate_channel is True:
        for i in range(img.shape[2]):
            new_min = random.randint(start_min_change, start_max_change)
            new_max = 255 - random.randint(end_min_change, end_max_change)
            if new_min != 0 or new_max != 255:
                img[:, :, i] = np.round(new_min + ((new_max - new_min)*img[:, :, i])/255)
    else:
        new_min = random.randint(start_min_change, start_max_change)
        new_max = 255 - random.randint(end_min_change, end_max_change)
        if new_min != 0 or new_max != 255:
            img = np.round(new_min + ((new_max - new_min)*img)/255)

    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


def normalize_brightness(img, separate_channel=False):
    if separate_channel is True:
        for i in range(img.shape[2]):
            mn = img[:, :, i].min()
            mx = img[:, :, i].max()
            if mn > 0 or mx < 255:
                img[:, :, i] = np.round(255*((img[:, :, i] - mn)/(mx - mn)))
    else:
        mn = img.min()
        mx = img.max()
        if mn > 0 or mx < 255:
            img = np.round(255 * ((img - mn) / (mx - mn)))

    return img.astype(np.uint8)


def random_intensity_change(img, min_change=0, max_change=20, separate_channel=False):
    img = img.astype(np.float32)
    delta = random.randint(min_change, max_change)
    for j in range(3):
        if separate_channel:
            delta = random.randint(min_change, max_change)
        img[:, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)


def random_rotate_with_mask(image, mask, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    dst_msk = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst, dst_msk


def random_rotate_with_mask_multichannel(image, mask, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    dst_msk = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        dst_msk[i] = cv2.warpAffine(mask[i], M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst, dst_msk


def random_mirror_with_mask(image, mask):
    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle).copy()
        mask = np.rot90(mask, k=angle).copy()
    return image, mask


def random_mirror_with_mask_multichannel(image, mask):
    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :].copy()
        mask = mask[:, ::-1, :].copy()
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle).copy()
        for i in range(mask.shape[0]):
            mask[i] = np.rot90(mask[i], k=angle).copy()
    # print(mirror, angle)
    return image, mask


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def save_history_figure(history, path, columns=('loss', 'val_loss')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()
