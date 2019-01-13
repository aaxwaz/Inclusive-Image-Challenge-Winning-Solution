# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 3
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


CACHE_PATH_TRAIN = OUTPUT_PATH + 'cache_resnet50_sh_336_train/'
if not os.path.isdir(CACHE_PATH_TRAIN):
    os.mkdir(CACHE_PATH_TRAIN)


PRECISION = 6
EPS = 0.0001


def validate_model(model_path, thr=0.5):
    from keras.models import load_model
    restore_from_cache = True
    box_size = 336
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
    model_path = MODELS_PATH + 'resnet50_336_temp.h5'
    thr = 0.5
    validate_model(model_path, thr)
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
F2 valid: 0.459475 [THR: 0.5] Time: 900 sec F2 test: 0.317676  [THR: 0.5] Time: 1932 sec LB: 0.314

Error in validation (values not equal 1 in confidence):
F2 valid: 0.472098 [THR: 0.2]
F2 valid: 0.478504 [THR: 0.25]
F2 valid: 0.482064 [THR: 0.3]
F2 valid: 0.482335 [THR: 0.35] F2 test: 0.324063 [THR: 0.35] LB: 0.313
F2 valid: 0.482026 [THR: 0.4] F2 test: 0.325847 [THR: 0.4]
F2 valid: 0.474908 [THR: 0.5] F2 test: 0.331256 [THR: 0.5] LB: 0.321
F2 test: 0.328287 [THR: 0.6]
F2 test: 0.331658 [THR: 0.55]
F2 valid: 0.444328 [THR: 0.7]
F2 test: 0.277767 [THR: 0.9]

New validation:
F2 valid: 0.682401 [THR: 0.1] F2 test: 0.256851 [THR: 0.1]

v2/resnet50_336_temp_146.h5
Valid: 0.6220 F2 test: 0.331256 [THR: 0.5]

v2/resnet50_336_temp_186.h5
Valid: 0.6475 F2 test: 0.327832 [THR: 0.5]

v2/resnet50_336_temp_190.h5
Valid: 0.6511 F2 test: 0.325867 [THR: 0.5]

v2/resnet50_336_temp_206.h5
Valid: 0.6610 F2 test: 0.335128 [THR: 0.5]

v2/resnet50_336_temp_221.h5
Valid: 0.6539 F2 test: 0.334925 [THR: 0.5]

v2/resnet50_336_temp_288.h5
F2 valid: 0.521756 [THR: 0.5] F2 test: 0.341477 [THR: 0.5]

v2/resnet50_336_temp_488.h5
F2 valid: 0.558833 [THR: 0.5] F2 test: 0.346290 [THR: 0.5]
F2 valid: 0.500088 [THR: 0.6] F2 test: 0.349956 [THR: 0.6] LB: 0.340
'''
