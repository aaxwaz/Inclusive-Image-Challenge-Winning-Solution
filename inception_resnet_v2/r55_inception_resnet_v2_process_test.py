# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a01_neural_nets import *
from a02_common_training_structures import *


PRECISION = 6
EPS = 0.00001


def process_test_images(model_path, box_size, cache_path_test):
    from keras.models import load_model

    if not os.path.isdir(cache_path_test):
        os.mkdir(cache_path_test)

    restore_from_cache = True
    model = load_model(model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})

    test_files = glob.glob(TEST_IMAGES_PATH + '*.jpg')
    print('Test valid:', len(test_files))

    for i, f in enumerate(sorted(test_files)):
        id = os.path.basename(f)[:-4]
        path = f

        cache_path = cache_path_test + id + '.pkl'
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
    model_path = MODELS_PATH + 'inception_resnet_v2_temp_320.h5'
    process_test_images(model_path, 299, OUTPUT_PATH + 'cache_inception_resnet_v2_test/')
    print('Time: {:.0f} sec'.format(time.time() - start_time))
