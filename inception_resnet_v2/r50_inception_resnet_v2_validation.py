# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from a00_common_functions import *
from a01_neural_nets import *
from neural_nets.a00_augmentation_functions import *
from a02_common_training_structures import *


CACHE_PATH_TRAIN = OUTPUT_PATH + 'cache_inception_resnet_v2_train/'
if not os.path.isdir(CACHE_PATH_TRAIN):
    os.mkdir(CACHE_PATH_TRAIN)


PRECISION = 6
EPS = 0.00001


def validate_model(model_path, thr=0.5):
    from keras.models import load_model
    restore_from_cache = True
    box_size = 299
    model = load_model(model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
    image_classes = get_classes_for_images_dict()
    train_files, valid_files = get_train_valid_split_of_files()
    print('Split files valid:', len(valid_files))
    score_avg = 0
    for i, id in enumerate(sorted(valid_files)):

        cache_path = CACHE_PATH_TRAIN + id + '.pkl'
        if not os.path.isfile(cache_path) or restore_from_cache is False:
            path = DATASET_PATH + 'train/' + id[:3] + '/' + id + '.jpg'
            im_full_big = read_single_image(path)
            im_full_big = cv2.resize(im_full_big, (box_size, box_size), cv2.INTER_LANCZOS4)

            batch_images = []
            batch_images.append(im_full_big.copy())
            batch_images.append(im_full_big[:, ::-1, :].copy())
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_images = preprocess_input(batch_images)

            preds = model.predict(batch_images)
            preds[preds < EPS] = 0
            preds = np.round(preds, PRECISION)
            save_in_file(preds, cache_path)
        else:
            preds = load_from_file(cache_path)

        batch_target = get_target_v2([id], image_classes).astype(np.uint8)
        preds = preds.mean(axis=0)
        # print(batch_target.shape)
        # print(preds.shape)
        preds[preds > thr] = 1
        preds[preds <= thr] = 0
        # print(preds.sum())
        score = fbeta_score(batch_target, np.expand_dims(preds, axis=0).astype(np.uint8), beta=2, average='samples')
        score_avg += score
        print('{} {}: {:.6f} {:.6f}'.format(i, id, score, score_avg/(i+1)))

    score_avg /= len(valid_files)
    print('F2 valid: {:.6f} [THR: {}]'.format(score_avg, thr))
    return score_avg


if __name__ == '__main__':
    start_time = time.time()
    model_path = MODELS_PATH + 'inception_resnet_v2_temp.h5'
    thr = 0.75
    validate_model(model_path, thr)
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
v1: inception_resnet_v2_temp_183.h5
F2 valid: 0.579960 [THR: 0.5]
F2 test: 0.354023 [THR: 0.5]
F2 test: 0.360603 [THR: 0.6] - mean of 2 predictions
F2 test: 0.360046 [THR: 0.6] - single prediction

v2: inception_resnet_v2_temp_320.h5
F2 valid: 0.728833 [THR: 0.2]
F2 valid: 0.538935 [THR: 0.6]
F2 valid: 0.475984 [THR: 0.7]
F2 valid: 0.400159 [THR: 0.8]

F2 test: 0.321955 [THR: 0.2]
F2 test: 0.363383 [THR: 0.6]
F2 test: 0.368020 [THR: 0.65]
F2 test: 0.368966 [THR: 0.7]
F2 test: 0.365990 [THR: 0.75]
F2 test: 0.361562 [THR: 0.8]
'''
