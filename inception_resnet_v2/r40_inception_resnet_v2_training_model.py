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
from albumentations import *
import random


def strong_aug(p=.5):
    return Compose([
        # RandomRotate90(),
        HorizontalFlip(),
        # Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.1),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        ToGray(p=0.05),
        JpegCompression(p=0.2, quality_lower=55, quality_upper=99),
        ElasticTransform(p=0.1),
    ], p=p)


global_aug = strong_aug(p=0.8)


def process_single_item(inp_file, box_size, augment=True):
    global global_aug
    f = DATASET_PATH + 'train/' + inp_file[:3] + '/' + inp_file + '.jpg'

    im_full_big = read_single_image(f)
    if im_full_big is None:
        im_full_big = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    CROP_SIZE = 10
    if augment is True or 0:
        # Random box (0-10% from each side)
        start_0 = random.randint(0, CROP_SIZE)
        start_1 = random.randint(0, CROP_SIZE)
        end_0 = im_full_big.shape[0] - random.randint(0, CROP_SIZE)
        end_1 = im_full_big.shape[1] - random.randint(0, CROP_SIZE)
        im_part = im_full_big[start_0:end_0, start_1:end_1, :]
    else:
        im_part = im_full_big

    im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)
    if augment is True:
        if 0:
            if random.randint(0, 1) == 0:
                # fliplr
                im_part = im_part[:, ::-1, :]
            im_part = random_intensity_change(im_part, -30, 30)
        im_part = global_aug(image=im_part)['image']

    return im_part


def get_target(batch_files, image_classes):
    target = np.zeros((len(batch_files), 7178), dtype=np.float32)
    for i, b in enumerate(batch_files):
        for el in image_classes[b]:
            target[i, el] = image_classes[b][el]
    return target


def batch_generator_train(files, image_classes, batch_size, augment=True):
    nn_shape = 299
    threads = 4

    # threads = 1
    p = ThreadPool(threads)
    process_item_func = partial(process_single_item, augment=augment, box_size=nn_shape)
    index = np.arange(0, files.shape[0])
    np.random.shuffle(index)

    start_index = 0
    while True:
        batch_index = index[start_index:start_index + batch_size]

        start_index += batch_size
        if start_index > len(files) - batch_size:
            start_index = 0

        batch_files = files[batch_index].copy()
        batch_target = get_target(batch_files, image_classes)
        batch_images = p.map(process_item_func, batch_files)
        batch_images = np.array(batch_images, np.float32)
        if 0:
            print(batch_images.shape, batch_target.shape)
            show_image(batch_images[0])
            print(batch_target[0], batch_target[0].sum())

        batch_images = preprocess_input(batch_images)
        yield batch_images, batch_target


def train_single_model(train_files, valid_files, epochs_num):
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from keras.optimizers import Adam, SGD
    from keras.losses import mean_squared_error
    from keras.models import load_model

    image_classes = get_classes_for_images_dict()

    restore = 1
    patience = 50
    epochs = epochs_num
    optim_type = 'Adam'
    learning_rate = 0.001
    cnn_type = 'inception_resnet_v2'
    print('Creating and compiling {}...'.format(cnn_type))

    tuning_test_images, tuning_test_labels = get_tuning_labels_data_for_validation(299)

    final_model_path = MODELS_PATH + '{}.h5'.format(cnn_type)
    if os.path.isfile(final_model_path) and restore == 0:
        print('Model already exists {}.'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp.h5'.format(cnn_type)
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        initial_epoch = 0
        model = load_model(cache_model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
    else:
        print('Start training from begining')
        initial_epoch = 0
        model = get_model_inception_resnet_v2()
        print(model.summary())

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, fbeta])
    # model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[dice_coef])

    print('Fitting model...')
    batch_size = 24
    print('Batch size: {}'.format(batch_size))
    steps_per_epoch = 48000 // batch_size
    validation_steps = len(valid_files) // (batch_size)
    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_fbeta', mode='max', save_best_only=True, verbose=0),
        # ModelCheckpoint(cache_model_path[:-3] + '_{epoch:02d}.h5', monitor='val_fbeta', mode='max', verbose=0),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_{}_lr_{}_optim_{}.csv'.format(cnn_type,
                                                                                       learning_rate,
                                                                                       optim_type), append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=1e-9, min_delta=0.00001, verbose=1, mode='min'),
        # CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=1000)
        ModelCheckpoint_F2Score(cache_model_path[:-3] + '_{epoch:02d}.h5', save_best_only=True,
                                mode='max', patience=patience, verbose=1,
                                validation_data=(tuning_test_images, tuning_test_labels)),
    ]

    history = model.fit_generator(generator=batch_generator_train(np.array(list(train_files)), image_classes, batch_size, True),
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator_train(np.array(list(valid_files)), image_classes, batch_size, False),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=10,
                                  initial_epoch=initial_epoch,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss: {} [Ep: {}]'.format(min_loss, len(history.history['val_loss'])))
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    save_history_figure(history, filename[:-4] + '.png')
    # del model
    # K.clear_session()
    return min_loss


def create_models_inception_resnet_v2(epochs_num):
    train_files, valid_files = get_train_valid_split_of_files()
    print('Split files train:', len(train_files))
    print('Split files valid:', len(valid_files))
    train_single_model(train_files, valid_files, epochs_num)


if __name__ == '__main__':
    start_time = time.time()
    global_aug = strong_aug(p=0.5)
    create_models_inception_resnet_v2(185)
    global_aug = strong_aug(p=0.8)
    create_models_inception_resnet_v2(135)
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
Ep 01: 1350s - loss: 0.0067 - f2beta_loss: -1.4234e-01 - fbeta: 0.1086 - val_loss: 0.0073 - val_f2beta_loss: -1.4875e-01 - val_fbeta: 0.1282 F2Beta score: 0.196525 THR: 0.3
Ep 02: 1224s - loss: 0.0048 - f2beta_loss: -2.1125e-01 - fbeta: 0.1974 - val_loss: 0.0064 - val_f2beta_loss: -1.8113e-01 - val_fbeta: 0.1720 F2Beta score: 0.173393 THR: 0.2
Ep 03: 1224s - loss: 0.0044 - f2beta_loss: -2.4482e-01 - fbeta: 0.2489 - val_loss: 0.0061 - val_f2beta_loss: -2.1110e-01 - val_fbeta: 0.2076 F2Beta score: 0.229967 THR: 0.3
Ep 04: 1221s - loss: 0.0041 - f2beta_loss: -2.7052e-01 - fbeta: 0.2885 - val_loss: 0.0057 - val_f2beta_loss: -2.4826e-01 - val_fbeta: 0.2696 F2Beta score: 0.238106 THR: 0.4
Ep 05: 1222s - loss: 0.0039 - f2beta_loss: -2.9110e-01 - fbeta: 0.3209 - val_loss: 0.0054 - val_f2beta_loss: -2.6311e-01 - val_fbeta: 0.3003 F2Beta score: 0.194693 THR: 0.3
Ep 06: 1221s - loss: 0.0037 - f2beta_loss: -3.0808e-01 - fbeta: 0.3478 - val_loss: 0.0054 - val_f2beta_loss: -2.6681e-01 - val_fbeta: 0.2945 F2Beta score: 0.228002 THR: 0.4
Ep 07: 1222s - loss: 0.0036 - f2beta_loss: -3.2514e-01 - fbeta: 0.3749 - val_loss: 0.0052 - val_f2beta_loss: -2.9570e-01 - val_fbeta: 0.3306 F2Beta score: 0.255973 THR: 0.4
Ep 19: 1228s - loss: 0.0029 - f2beta_loss: -4.1457e-01 - fbeta: 0.5166 - val_loss: 0.0041 - val_f2beta_loss: -3.8507e-01 - val_fbeta: 0.4800 F2Beta score: 0.278956 THR: 0.4
Ep 33: 1226s - loss: 0.0026 - f2beta_loss: -4.5523e-01 - fbeta: 0.5770 - val_loss: 0.0036 - val_f2beta_loss: -4.3356e-01 - val_fbeta: 0.5489 F2Beta score: 0.315234 THR: 0.5
Ep 39: 1229s - loss: 0.0025 - f2beta_loss: -4.6711e-01 - fbeta: 0.5952 - val_loss: 0.0035 - val_f2beta_loss: -4.4128e-01 - val_fbeta: 0.5529 F2Beta score: 0.321912 THR: 0.5
Ep 44: 1224s - loss: 0.0025 - f2beta_loss: -4.7679e-01 - fbeta: 0.6085 - val_loss: 0.0034 - val_f2beta_loss: -4.6668e-01 - val_fbeta: 0.5952 F2Beta score: 0.329928 THR: 0.5
Ep 49: 1246s - loss: 0.0024 - f2beta_loss: -4.8252e-01 - fbeta: 0.6177 - val_loss: 0.0034 - val_f2beta_loss: -4.6660e-01 - val_fbeta: 0.5857 F2Beta score: 0.308434 THR: 0.5
Ep 66: 1260s - loss: 0.0022 - f2beta_loss: -5.0972e-01 - fbeta: 0.6570 - val_loss: 0.0031 - val_f2beta_loss: -4.8658e-01 - val_fbeta: 0.6137 F2Beta score: 0.338490 THR: 0.5
Ep 70: 1265s - loss: 0.0022 - f2beta_loss: -5.1048e-01 - fbeta: 0.6579 - val_loss: 0.0031 - val_f2beta_loss: -5.0070e-01 - val_fbeta: 0.6290 F2Beta score: 0.340333 THR: 0.5
Ep 75: 1276s - loss: 0.0022 - f2beta_loss: -5.1676e-01 - fbeta: 0.6665 - val_loss: 0.0031 - val_f2beta_loss: -5.0537e-01 - val_fbeta: 0.6367 F2Beta score: 0.342857 THR: 0.6
Ep 80: 1285s - loss: 0.0022 - f2beta_loss: -5.2183e-01 - fbeta: 0.6738 - val_loss: 0.0031 - val_f2beta_loss: -5.1275e-01 - val_fbeta: 0.6484 F2Beta score: 0.344861 THR: 0.6
Ep 82: 1295s - loss: 0.0022 - f2beta_loss: -5.2245e-01 - fbeta: 0.6742 - val_loss: 0.0030 - val_f2beta_loss: -5.0022e-01 - val_fbeta: 0.6304 F2Beta score: 0.346157 THR: 0.6
Ep 84: 1223s - loss: 0.0021 - f2beta_loss: -5.2990e-01 - fbeta: 0.6846 - val_loss: 0.0030 - val_f2beta_loss: -5.0760e-01 - val_fbeta: 0.6358 F2Beta score: 0.344545 THR: 0.6
Ep 85: 1223s - loss: 0.0021 - f2beta_loss: -5.2893e-01 - fbeta: 0.6825 - val_loss: 0.0030 - val_f2beta_loss: -5.0960e-01 - val_fbeta: 0.6407 F2Beta score: 0.347497 THR: 0.6
Ep 88: 1225s - loss: 0.0021 - f2beta_loss: -5.3071e-01 - fbeta: 0.6846 - val_loss: 0.0030 - val_f2beta_loss: -5.1217e-01 - val_fbeta: 0.6415 F2Beta score: 0.349612 THR: 0.6
Ep 95: 1223s - loss: 0.0021 - f2beta_loss: -5.3282e-01 - fbeta: 0.6884 - val_loss: 0.0030 - val_f2beta_loss: -5.2368e-01 - val_fbeta: 0.6611 F2Beta score: 0.351130 THR: 0.6
Ep 98: 1222s - loss: 0.0021 - f2beta_loss: -5.3472e-01 - fbeta: 0.6904 - val_loss: 0.0029 - val_f2beta_loss: -5.2353e-01 - val_fbeta: 0.6583 F2Beta score: 0.351727 THR: 0.6
Ep 120:1222s - loss: 0.0019 - f2beta_loss: -5.5674e-01 - fbeta: 0.7224 - val_loss: 0.0029 - val_f2beta_loss: -5.4139e-01 - val_fbeta: 0.6798 F2Beta score: 0.354428 THR: 0.7
Ep 130:1226s - loss: 0.0019 - f2beta_loss: -5.5855e-01 - fbeta: 0.7248 - val_loss: 0.0028 - val_f2beta_loss: -5.3722e-01 - val_fbeta: 0.6751 F2Beta score: 0.360496 THR: 0.6
Ep 136:1238s - loss: 0.0019 - f2beta_loss: -5.6111e-01 - fbeta: 0.7271 - val_loss: 0.0028 - val_f2beta_loss: -5.4137e-01 - val_fbeta: 0.6788 F2Beta score: 0.360656 THR: 0.6
Ep 138:1240s - loss: 0.0019 - f2beta_loss: -5.6259e-01 - fbeta: 0.7286 - val_loss: 0.0028 - val_f2beta_loss: -5.4108e-01 - val_fbeta: 0.6793 F2Beta score: 0.361576 THR: 0.6
Ep 147:1223s - loss: 0.0019 - f2beta_loss: -5.6354e-01 - fbeta: 0.7311 - val_loss: 0.0028 - val_f2beta_loss: -5.4503e-01 - val_fbeta: 0.6843 F2Beta score: 0.359643 THR: 0.6
Ep 159:1225s - loss: 0.0019 - f2beta_loss: -5.6679e-01 - fbeta: 0.7340 - val_loss: 0.0028 - val_f2beta_loss: -5.4774e-01 - val_fbeta: 0.6859 F2Beta score: 0.364725 THR: 0.7
Ep 183:1222s - loss: 0.0018 - f2beta_loss: -5.7441e-01 - fbeta: 0.7454 - val_loss: 0.0027 - val_f2beta_loss: -5.5239e-01 - val_fbeta: 0.6921 F2Beta score: 0.365691 THR: 0.7
Restart here with augm 0.8 and Adam = 0.001 again
Ep 187:1224s - loss: 0.0022 - f2beta_loss: -5.1509e-01 - fbeta: 0.6604 - val_loss: 0.0030 - val_f2beta_loss: -5.0538e-01 - val_fbeta: 0.6333 F2Beta score: 0.350152 THR: 0.5
Ep 202:1225s - loss: 0.0022 - f2beta_loss: -5.2323e-01 - fbeta: 0.6716 - val_loss: 0.0029 - val_f2beta_loss: -5.2599e-01 - val_fbeta: 0.6592 F2Beta score: 0.357315 THR: 0.6
Ep 238:1240s - loss: 0.0020 - f2beta_loss: -5.4209e-01 - fbeta: 0.6980 - val_loss: 0.0028 - val_f2beta_loss: -5.4190e-01 - val_fbeta: 0.6762 F2Beta score: 0.364095 THR: 0.6
Ep 248:1254s - loss: 0.0020 - f2beta_loss: -5.4316e-01 - fbeta: 0.7000 - val_loss: 0.0028 - val_f2beta_loss: -5.4019e-01 - val_fbeta: 0.6756 F2Beta score: 0.365846 THR: 0.6
Ep 255:1329s - loss: 0.0020 - f2beta_loss: -5.4616e-01 - fbeta: 0.7049 - val_loss: 0.0028 - val_f2beta_loss: -5.5036e-01 - val_fbeta: 0.6847 F2Beta score: 0.367385 THR: 0.7
Ep 281:1336s - loss: 0.0019 - f2beta_loss: -5.5784e-01 - fbeta: 0.7205 - val_loss: 0.0027 - val_f2beta_loss: -5.5229e-01 - val_fbeta: 0.6889 F2Beta score: 0.371077 THR: 0.6
Ep 293:1362s - loss: 0.0019 - f2beta_loss: -5.6466e-01 - fbeta: 0.7301 - val_loss: 0.0027 - val_f2beta_loss: -5.5729e-01 - val_fbeta: 0.6921 F2Beta score: 0.371170 THR: 0.7
Ep 320:1490s - loss: 0.0019 - f2beta_loss: -5.6904e-01 - fbeta: 0.7356 - val_loss: 0.0027 - val_f2beta_loss: -5.6610e-01 - val_fbeta: 0.7038 F2Beta score: 0.373804 THR: 0.7
'''
