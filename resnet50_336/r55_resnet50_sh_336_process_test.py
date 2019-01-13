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


CACHE_PATH_TEST = OUTPUT_PATH + 'cache_resnet50_sh_336_test/'
if not os.path.isdir(CACHE_PATH_TEST):
    os.mkdir(CACHE_PATH_TEST)


PRECISION = 6
EPS = 0.0001


def process_test(model_path):
    from keras.models import load_model
    box_size = 336
    restore_from_cache = True
    model = load_model(model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
    test_files = glob.glob(TEST_IMAGES_PATH + '*.jpg')
    print('Test valid:', len(test_files))

    for i, f in enumerate(sorted(test_files)):
        id = os.path.basename(f)[:-4]
        path = f

        cache_path = CACHE_PATH_TEST + id + '.pkl'
        if not os.path.isfile(cache_path) or restore_from_cache is False:
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


if __name__ == '__main__':
    start_time = time.time()
    model_path = MODELS_PATH + 'resnet50_336_temp_488.h5'
    process_test(model_path)
    print('Time: {:.0f} sec'.format(time.time() - start_time))
