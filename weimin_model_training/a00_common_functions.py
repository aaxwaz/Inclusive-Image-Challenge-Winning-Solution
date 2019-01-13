import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
import random
import shutil
import operator
#import pyvips
#from PIL import Image
import platform
import json
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
from keras import backend as K

############# SET PATHS ################
DATASET_PATH = '/mnt/YTdata/inclusive/train_images/'
ROOT_PATH = '/home/weimin/reproduce/2018_09_Inclusive_Images_submit_final/'
TUNING_IMAGE_PATH = '/home/weimin/reproduce/2018_09_Inclusive_Images_submit_final/input/tuning_label_images/*.jpg'
#######################################

INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
VAL_DATAPATH = ROOT_PATH + 'val_images/'

MODELS_PATH = ROOT_PATH + 'models/'
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
SUBM_PATH = ROOT_PATH + 'subm/'

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
if not os.path.isdir(VAL_DATAPATH):
    os.mkdir(VAL_DATAPATH)
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))

def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('fbeta', 'val_fbeta')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()

def load_img_npz(path):
    """Loads an image from pickle file.
    # Arguments
        path: Path to image file
    # Returns
        An image instance
    """
    #img = load_from_file_fast(path)

    temp_img = cv2.imread(path)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
    if temp_img.shape[:2] != (299,299):
        temp_img = cv2.resize(temp_img, (299,299))

    return temp_img.astype(np.float32)
    
def read_single_image(path):

    #try:
    #    img = np.array(Image.open(path))
    #except:
    try:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    except:
        print('Fail to read image path: ', path)
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def read_single_image_keras(path):

    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img_to_array(img)
    return img

def get_description_for_labels():
    out = open(INPUT_PATH + 'class-descriptions-boxable.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def get_subcategories(sub_cat, upper_cat, level, l, d1, sub):
    ret = []
    sub_cat[upper_cat] = ([], [])
    for j, k in enumerate(l[sub]):
        nm = d1[k['LabelName']]
        sub_cat[upper_cat][1].append(nm)
        if nm in sub_cat:
            continue
        ret.append(nm)
        if 'Subcategory' in k:
            get_subcategories(sub_cat, nm, level + 1, l, d1, 'Subcategory')
        else:
            sub_cat[nm] = ([upper_cat], [])
    return ret


def get_hierarchy_structures():
    sub_cat = dict()
    part_cat = dict()
    d1, d2 = get_description_for_labels()
    arr = json.load(open(INPUT_PATH + 'bbox_labels_600_hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    for i, l in enumerate(lst):
        nm = d1[l['LabelName']]
        if 'Subcategory' in l:
            get_subcategories(sub_cat, nm, 1, l, d1, 'Subcategory')
        else:
            if nm in sub_cat:
                print('Strange!')
                exit()
            sub_cat[nm] = [], []
    return sub_cat


def get_description_for_labels_500():
    out = open(INPUT_PATH + 'challenge-2018-class-descriptions-500.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def get_classes_dict(type='all'):
    dcode, dreal = dict(), dict()
    s1 = list(pd.read_csv(INPUT_PATH + 'classes-trainable.csv')['label_code'].values)
    s2 = pd.read_csv(INPUT_PATH + 'class-descriptions.csv')
    orig = s2['label_code'].values
    descr = s2['description'].values

    if type == 'all':
        for i, o in enumerate(orig):
            dcode[o] = descr[i]
            dreal[descr[i]] = o
    else:
        for i, o in enumerate(orig):
            if o in s1:
                dcode[o] = descr[i]
                dreal[descr[i]] = o

    return dcode, dreal

def get_classes_dict_top_543(type='all'):
    dcode, dreal = dict(), dict()
    s1 = set(pd.read_csv(INPUT_PATH + 'final_label_543_count_500.csv')['LabelName'].values)
    s2 = pd.read_csv(INPUT_PATH + 'class-descriptions.csv')
    orig = s2['label_code'].values
    descr = s2['description'].values

    if type == 'all':
        for i, o in enumerate(orig):
            dcode[o] = descr[i]
            dreal[descr[i]] = o
    else:
        for i, o in enumerate(orig):
            if o in s1:
                dcode[o] = descr[i]
                dreal[descr[i]] = o

    return dcode, dreal


def get_classes_dict_top_491(type='all'):
    dcode, dreal = dict(), dict()
    s1 = set(pd.read_csv(INPUT_PATH + 'new_labels_from_test.csv')['labels'].values)
    s2 = pd.read_csv(INPUT_PATH + 'class-descriptions.csv')
    orig = s2['label_code'].values
    descr = s2['description'].values

    if type == 'all':
        for i, o in enumerate(orig):
            dcode[o] = descr[i]
            dreal[descr[i]] = o
    else:
        for i, o in enumerate(orig):
            if o in s1:
                dcode[o] = descr[i]
                dreal[descr[i]] = o

    return dcode, dreal

def get_classes_to_index_dicts():
    s1 = list(pd.read_csv(INPUT_PATH + 'classes-trainable.csv')['label_code'].values)
    index_arr_forward = dict()
    index_arr_backward = dict()
    for i in range(len(s1)):
        index_arr_forward[s1[i]] = i
        index_arr_backward[i] = s1[i]

    return index_arr_forward, index_arr_backward

def get_classes_to_index_dicts_top_543():
    s1 = list(pd.read_csv(INPUT_PATH + 'final_label_543_count_500.csv')['LabelName'].values)
    index_arr_forward = dict()
    index_arr_backward = dict()
    for i in range(len(s1)):
        index_arr_forward[s1[i]] = i
        index_arr_backward[i] = s1[i]

    return index_arr_forward, index_arr_backward

def get_classes_to_index_dicts_top_491():
    s1 = list(pd.read_csv(INPUT_PATH + 'new_labels_from_test.csv')['labels'].values)
    index_arr_forward = dict()
    index_arr_backward = dict()
    for i in range(len(s1)):
        index_arr_forward[s1[i]] = i
        index_arr_backward[i] = s1[i]

    return index_arr_forward, index_arr_backward

def get_target_v2(batch_files, image_classes):
    target = np.zeros((len(batch_files), 7178), dtype=np.uint8)
    for i, b in enumerate(batch_files):
        for el in image_classes[b]:
            target[i, el] = 1
    return target

def get_target_top_491(batch_files, image_classes):
    target = np.zeros((len(batch_files), 491), dtype=np.uint8)
    for i, b in enumerate(batch_files):
        for el in image_classes[b]:
            target[i, el] = 1
    return target
