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
    nn_shape = 336
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


def train_single_model(train_files, valid_files):
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from keras.optimizers import Adam, SGD
    from keras.losses import mean_squared_error
    from keras.models import load_model

    image_classes = get_classes_for_images_dict()

    restore = 1
    patience = 50
    epochs = 488
    optim_type = 'Adam'
    learning_rate = 0.0001
    cnn_type = 'resnet50_336'
    print('Creating and compiling {}...'.format(cnn_type))

    tuning_test_images, tuning_test_labels = get_tuning_labels_data_for_validation(336)

    final_model_path = MODELS_PATH + '{}.h5'.format(cnn_type)
    if os.path.isfile(final_model_path) and restore == 0:
        print('Model already exists {}.'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp.h5'.format(cnn_type)
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model = load_model(cache_model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
    else:
        print('Start training from begining')
        model = get_model_resnet50_336()
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
                                  initial_epoch=0,
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


def create_models_resnet50():
    train_files, valid_files = get_train_valid_split_of_files()
    print('Split files train:', len(train_files))
    print('Split files valid:', len(valid_files))
    train_single_model(train_files, valid_files)


if __name__ == '__main__':
    start_time = time.time()
    create_models_resnet50()
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
833/833 [==============================] - 494s 593ms/step - loss: 0.0134 - f2beta_loss: -0.0871 - fbeta: 0.0379 - val_loss: 0.0080 - val_f2beta_loss: -0.0945 - val_fbeta: 0.0645
1024s - loss: 0.0055 - f2beta_loss: -1.4412e-01 - fbeta: 0.0989 - val_loss: 0.0074 - val_f2beta_loss: -1.2535e-01 - val_fbeta: 0.0824
1005s - loss: 0.0049 - f2beta_loss: -1.8765e-01 - fbeta: 0.1548 - val_loss: 0.0067 - val_f2beta_loss: -1.6487e-01 - val_fbeta: 0.1406
1003s - loss: 0.0047 - f2beta_loss: -2.1159e-01 - fbeta: 0.1904 - val_loss: 0.0065 - val_f2beta_loss: -1.7862e-01 - val_fbeta: 0.1592
1002s - loss: 0.0044 - f2beta_loss: -2.3177e-01 - fbeta: 0.2206 - val_loss: 0.0063 - val_f2beta_loss: -1.7896e-01 - val_fbeta: 0.1611
1004s - loss: 0.0042 - f2beta_loss: -2.4741e-01 - fbeta: 0.2457 - val_loss: 0.0062 - val_f2beta_loss: -2.0656e-01 - val_fbeta: 0.2041
1005s - loss: 0.0041 - f2beta_loss: -2.5963e-01 - fbeta: 0.2636 - val_loss: 0.0061 - val_f2beta_loss: -2.1336e-01 - val_fbeta: 0.2134
 998s - loss: 0.0040 - f2beta_loss: -2.7052e-01 - fbeta: 0.2846 - val_loss: 0.0056 - val_f2beta_loss: -2.4400e-01 - val_fbeta: 0.2624
 999s - loss: 0.0039 - f2beta_loss: -2.8241e-01 - fbeta: 0.3040 - val_loss: 0.0056 - val_f2beta_loss: -2.6355e-01 - val_fbeta: 0.2839
 988s - loss: 0.0038 - f2beta_loss: -2.9202e-01 - fbeta: 0.3199 - val_loss: 0.0055 - val_f2beta_loss: -2.5575e-01 - val_fbeta: 0.2835
1001s - loss: 0.0037 - f2beta_loss: -3.0036e-01 - fbeta: 0.3331 - val_loss: 0.0056 - val_f2beta_loss: -2.4589e-01 - val_fbeta: 0.2572
 934s - loss: 0.0038 - f2beta_loss: -2.9041e-01 - fbeta: 0.3151 - val_loss: 0.0056 - val_f2beta_loss: -2.3635e-01 - val_fbeta: 0.2554
 931s - loss: 0.0038 - f2beta_loss: -2.9756e-01 - fbeta: 0.3283 - val_loss: 0.0055 - val_f2beta_loss: -2.5320e-01 - val_fbeta: 0.2686
 933s - loss: 0.0037 - f2beta_loss: -3.0435e-01 - fbeta: 0.3405 - val_loss: 0.0053 - val_f2beta_loss: -2.7352e-01 - val_fbeta: 0.3178
 915s - loss: 0.0037 - f2beta_loss: -3.1154e-01 - fbeta: 0.3526 - val_loss: 0.0052 - val_f2beta_loss: -2.7143e-01 - val_fbeta: 0.3023
 915s - loss: 0.0036 - f2beta_loss: -3.1765e-01 - fbeta: 0.3615 - val_loss: 0.0052 - val_f2beta_loss: -2.9125e-01 - val_fbeta: 0.3370
 Ep 06: 913s - loss: 0.0036 - f2beta_loss: -3.2253e-01 - fbeta: 0.3711 - val_loss: 0.0050 - val_f2beta_loss: -2.9943e-01 - val_fbeta: 0.3479
 Ep 07: 914s - loss: 0.0035 - f2beta_loss: -3.2609e-01 - fbeta: 0.3783 - val_loss: 0.0053 - val_f2beta_loss: -2.8816e-01 - val_fbeta: 0.3289
 Ep 08: 910s - loss: 0.0035 - f2beta_loss: -3.3204e-01 - fbeta: 0.3852 - val_loss: 0.0050 - val_f2beta_loss: -2.9637e-01 - val_fbeta: 0.3335
 Ep 09: 920s - loss: 0.0034 - f2beta_loss: -3.3728e-01 - fbeta: 0.3946 - val_loss: 0.0049 - val_f2beta_loss: -2.9744e-01 - val_fbeta: 0.3431
 Ep 10: 919s - loss: 0.0034 - f2beta_loss: -3.4109e-01 - fbeta: 0.4006 - val_loss: 0.0049 - val_f2beta_loss: -3.0128e-01 - val_fbeta: 0.3465
 Ep 11: 919s - loss: 0.0033 - f2beta_loss: -3.4640e-01 - fbeta: 0.4104 - val_loss: 0.0049 - val_f2beta_loss: -3.0433e-01 - val_fbeta: 0.3467
 Ep 12: 920s - loss: 0.0033 - f2beta_loss: -3.5102e-01 - fbeta: 0.4153 - val_loss: 0.0050 - val_f2beta_loss: -3.1301e-01 - val_fbeta: 0.3688
 Ep 13: 925s - loss: 0.0033 - f2beta_loss: -3.5464e-01 - fbeta: 0.4225 - val_loss: 0.0048 - val_f2beta_loss: -3.1030e-01 - val_fbeta: 0.3560 
 Ep 14: 932s - loss: 0.0033 - f2beta_loss: -3.5868e-01 - fbeta: 0.4281 - val_loss: 0.0047 - val_f2beta_loss: -3.2805e-01 - val_fbeta: 0.3914
 Ep 15: 936s - loss: 0.0032 - f2beta_loss: -3.6224e-01 - fbeta: 0.4354 - val_loss: 0.0047 - val_f2beta_loss: -3.1859e-01 - val_fbeta: 0.3653 
 Ep 16: 923s - loss: 0.0032 - f2beta_loss: -3.6359e-01 - fbeta: 0.4387 - val_loss: 0.0048 - val_f2beta_loss: -3.2096e-01 - val_fbeta: 0.3802
 Ep 17: 916s - loss: 0.0032 - f2beta_loss: -3.6774e-01 - fbeta: 0.4451 - val_loss: 0.0045 - val_f2beta_loss: -3.3043e-01 - val_fbeta: 0.3971
 Ep 18: 915s - loss: 0.0032 - f2beta_loss: -3.7050e-01 - fbeta: 0.4491 - val_loss: 0.0046 - val_f2beta_loss: -3.1703e-01 - val_fbeta: 0.3715
 Ep 19: 923s - loss: 0.0031 - f2beta_loss: -3.7424e-01 - fbeta: 0.4541 - val_loss: 0.0046 - val_f2beta_loss: -3.3903e-01 - val_fbeta: 0.3958
 Ep 20: 923s - loss: 0.0031 - f2beta_loss: -3.7616e-01 - fbeta: 0.4573 - val_loss: 0.0045 - val_f2beta_loss: -3.4311e-01 - val_fbeta: 0.4136
 Ep 21: 923s - loss: 0.0031 - f2beta_loss: -3.8041e-01 - fbeta: 0.4656 - val_loss: 0.0045 - val_f2beta_loss: -3.4063e-01 - val_fbeta: 0.4133 
 Ep 22: 924s - loss: 0.0031 - f2beta_loss: -3.8037e-01 - fbeta: 0.4650 - val_loss: 0.0048 - val_f2beta_loss: -3.2676e-01 - val_fbeta: 0.3871
 Ep 23: 925s - loss: 0.0030 - f2beta_loss: -3.8714e-01 - fbeta: 0.4760 - val_loss: 0.0043 - val_f2beta_loss: -3.5813e-01 - val_fbeta: 0.4396 
 Ep 24: 935s - loss: 0.0031 - f2beta_loss: -3.8697e-01 - fbeta: 0.4745 - val_loss: 0.0044 - val_f2beta_loss: -3.3880e-01 - val_fbeta: 0.4090
 Ep 25: 940s - loss: 0.0030 - f2beta_loss: -3.8940e-01 - fbeta: 0.4788 - val_loss: 0.0045 - val_f2beta_loss: -3.4190e-01 - val_fbeta: 0.4039
 Ep 26: 943s - loss: 0.0030 - f2beta_loss: -3.9128e-01 - fbeta: 0.4818 - val_loss: 0.0044 - val_f2beta_loss: -3.5271e-01 - val_fbeta: 0.4280
 Ep 27: 935s - loss: 0.0030 - f2beta_loss: -3.9290e-01 - fbeta: 0.4853 - val_loss: 0.0044 - val_f2beta_loss: -3.5458e-01 - val_fbeta: 0.4251
 Ep 28: 938s - loss: 0.0030 - f2beta_loss: -3.9620e-01 - fbeta: 0.4901 - val_loss: 0.0042 - val_f2beta_loss: -3.6069e-01 - val_fbeta: 0.4456
 Ep 29: 939s - loss: 0.0029 - f2beta_loss: -3.9685e-01 - fbeta: 0.4907 - val_loss: 0.0042 - val_f2beta_loss: -3.5785e-01 - val_fbeta: 0.4334
Ep 30: 939s - loss: 0.0029 - f2beta_loss: -3.9979e-01 - fbeta: 0.4945 - val_loss: 0.0043 - val_f2beta_loss: -3.6427e-01 - val_fbeta: 0.4461
Ep 31: 959s - loss: 0.0029 - f2beta_loss: -4.0278e-01 - fbeta: 0.5005 - val_loss: 0.0042 - val_f2beta_loss: -3.6207e-01 - val_fbeta: 0.4429
Ep 32: 955s - loss: 0.0029 - f2beta_loss: -4.0330e-01 - fbeta: 0.5009 - val_loss: 0.0042 - val_f2beta_loss: -3.6132e-01 - val_fbeta: 0.4332
Ep 33: 943s - loss: 0.0029 - f2beta_loss: -4.0586e-01 - fbeta: 0.5059 - val_loss: 0.0042 - val_f2beta_loss: -3.8219e-01 - val_fbeta: 0.4709
Ep 34: 940s - loss: 0.0029 - f2beta_loss: -4.0804e-01 - fbeta: 0.5091 - val_loss: 0.0041 - val_f2beta_loss: -3.7422e-01 - val_fbeta: 0.4603
Ep 35: 944s - loss: 0.0029 - f2beta_loss: -4.0926e-01 - fbeta: 0.5097 - val_loss: 0.0040 - val_f2beta_loss: -3.8007e-01 - val_fbeta: 0.4695
Ep 36: 933s - loss: 0.0029 - f2beta_loss: -4.1177e-01 - fbeta: 0.5133 - val_loss: 0.0041 - val_f2beta_loss: -3.7932e-01 - val_fbeta: 0.4686
Ep 37: 949s - loss: 0.0028 - f2beta_loss: -4.1407e-01 - fbeta: 0.5158 - val_loss: 0.0041 - val_f2beta_loss: -3.7339e-01 - val_fbeta: 0.4617
Ep 38: 957s - loss: 0.0028 - f2beta_loss: -4.1631e-01 - fbeta: 0.5214 - val_loss: 0.0041 - val_f2beta_loss: -3.7705e-01 - val_fbeta: 0.4629
Ep 39: 957s - loss: 0.0028 - f2beta_loss: -4.1810e-01 - fbeta: 0.5241 - val_loss: 0.0040 - val_f2beta_loss: -3.8399e-01 - val_fbeta: 0.4774
Ep 40: 957s - loss: 0.0028 - f2beta_loss: -4.1911e-01 - fbeta: 0.5257 - val_loss: 0.0041 - val_f2beta_loss: -3.8349e-01 - val_fbeta: 0.4653
Ep 41: 959s - loss: 0.0028 - f2beta_loss: -4.2123e-01 - fbeta: 0.5280 - val_loss: 0.0041 - val_f2beta_loss: -3.8284e-01 - val_fbeta: 0.4718
Ep 42: 963s - loss: 0.0028 - f2beta_loss: -4.2276e-01 - fbeta: 0.5318 - val_loss: 0.0040 - val_f2beta_loss: -3.8823e-01 - val_fbeta: 0.4811
Ep 43: 962s - loss: 0.0028 - f2beta_loss: -4.2265e-01 - fbeta: 0.5324 - val_loss: 0.0040 - val_f2beta_loss: -3.8837e-01 - val_fbeta: 0.4786
Ep 44: 954s - loss: 0.0028 - f2beta_loss: -4.2572e-01 - fbeta: 0.5350 - val_loss: 0.0040 - val_f2beta_loss: -3.9217e-01 - val_fbeta: 0.4813
Ep 45: 951s - loss: 0.0027 - f2beta_loss: -4.2616e-01 - fbeta: 0.5369 - val_loss: 0.0039 - val_f2beta_loss: -3.8217e-01 - val_fbeta: 0.4711
Ep 46: 947s - loss: 0.0027 - f2beta_loss: -4.2750e-01 - fbeta: 0.5384 - val_loss: 0.0039 - val_f2beta_loss: -3.9812e-01 - val_fbeta: 0.4950
Ep 47: 967s - loss: 0.0027 - f2beta_loss: -4.2949e-01 - fbeta: 0.5421 - val_loss: 0.0040 - val_f2beta_loss: -3.9092e-01 - val_fbeta: 0.4804
Ep 48: 968s - loss: 0.0027 - f2beta_loss: -4.3112e-01 - fbeta: 0.5432 - val_loss: 0.0039 - val_f2beta_loss: -4.0759e-01 - val_fbeta: 0.5060
Ep 49: 969s - loss: 0.0027 - f2beta_loss: -4.3359e-01 - fbeta: 0.5483 - val_loss: 0.0039 - val_f2beta_loss: -4.1050e-01 - val_fbeta: 0.5054
Ep 50: 970s - loss: 0.0027 - f2beta_loss: -4.3416e-01 - fbeta: 0.5479 - val_loss: 0.0039 - val_f2beta_loss: -4.1198e-01 - val_fbeta: 0.5134
Ep 51: 970s - loss: 0.0027 - f2beta_loss: -4.3592e-01 - fbeta: 0.5523 - val_loss: 0.0039 - val_f2beta_loss: -4.0809e-01 - val_fbeta: 0.5075
Ep 52: 972s - loss: 0.0027 - f2beta_loss: -4.3474e-01 - fbeta: 0.5503 - val_loss: 0.0040 - val_f2beta_loss: -4.0110e-01 - val_fbeta: 0.4908
Ep 53: 976s - loss: 0.0027 - f2beta_loss: -4.3745e-01 - fbeta: 0.5548 - val_loss: 0.0038 - val_f2beta_loss: -3.9885e-01 - val_fbeta: 0.5053
Ep 54: 979s - loss: 0.0027 - f2beta_loss: -4.3842e-01 - fbeta: 0.5560 - val_loss: 0.0039 - val_f2beta_loss: -4.0824e-01 - val_fbeta: 0.5072
Ep 55: 981s - loss: 0.0026 - f2beta_loss: -4.4054e-01 - fbeta: 0.5590 - val_loss: 0.0039 - val_f2beta_loss: -4.1027e-01 - val_fbeta: 0.5000
Ep 56: 975s - loss: 0.0027 - f2beta_loss: -4.4055e-01 - fbeta: 0.5588 - val_loss: 0.0038 - val_f2beta_loss: -4.1834e-01 - val_fbeta: 0.5236
Ep 57: 984s - loss: 0.0026 - f2beta_loss: -4.4289e-01 - fbeta: 0.5633 - val_loss: 0.0038 - val_f2beta_loss: -3.9609e-01 - val_fbeta: 0.4922
Ep 58: 985s - loss: 0.0026 - f2beta_loss: -4.4124e-01 - fbeta: 0.5603 - val_loss: 0.0039 - val_f2beta_loss: -3.9914e-01 - val_fbeta: 0.4908
Ep 59: 991s - loss: 0.0026 - f2beta_loss: -4.4732e-01 - fbeta: 0.5685 - val_loss: 0.0039 - val_f2beta_loss: -4.0578e-01 - val_fbeta: 0.4999
Ep 60: 992s - loss: 0.0026 - f2beta_loss: -4.4589e-01 - fbeta: 0.5661 - val_loss: 0.0039 - val_f2beta_loss: -4.1509e-01 - val_fbeta: 0.5120
Ep 61: 997s - loss: 0.0026 - f2beta_loss: -4.4663e-01 - fbeta: 0.5677 - val_loss: 0.0038 - val_f2beta_loss: -4.1657e-01 - val_fbeta: 0.5159
Ep 62: 999s - loss: 0.0026 - f2beta_loss: -4.4792e-01 - fbeta: 0.5705 - val_loss: 0.0038 - val_f2beta_loss: -4.2387e-01 - val_fbeta: 0.5258
Ep 63: 1005s - loss: 0.0026 - f2beta_loss: -4.4734e-01 - fbeta: 0.5698 - val_loss: 0.0038 - val_f2beta_loss: -4.1994e-01 - val_fbeta: 0.5237
Ep 64: 984s - loss: 0.0026 - f2beta_loss: -4.4997e-01 - fbeta: 0.5733 - val_loss: 0.0037 - val_f2beta_loss: -4.2441e-01 - val_fbeta: 0.5359
Ep 65: 989s - loss: 0.0026 - f2beta_loss: -4.4958e-01 - fbeta: 0.5720 - val_loss: 0.0037 - val_f2beta_loss: -4.1358e-01 - val_fbeta: 0.5162
Ep 66: 1002s - loss: 0.0026 - f2beta_loss: -4.5109e-01 - fbeta: 0.5748 - val_loss: 0.0037 - val_f2beta_loss: -4.0932e-01 - val_fbeta: 0.5144
Ep 67: 1003s - loss: 0.0026 - f2beta_loss: -4.5298e-01 - fbeta: 0.5782 - val_loss: 0.0037 - val_f2beta_loss: -4.2437e-01 - val_fbeta: 0.5282
Ep 68: 1003s - loss: 0.0026 - f2beta_loss: -4.5263e-01 - fbeta: 0.5776 - val_loss: 0.0037 - val_f2beta_loss: -4.2289e-01 - val_fbeta: 0.5297
Ep 69: 1003s - loss: 0.0025 - f2beta_loss: -4.5455e-01 - fbeta: 0.5810 - val_loss: 0.0037 - val_f2beta_loss: -4.4039e-01 - val_fbeta: 0.5516
Ep 70: 1008s - loss: 0.0025 - f2beta_loss: -4.5578e-01 - fbeta: 0.5820 - val_loss: 0.0037 - val_f2beta_loss: -4.3438e-01 - val_fbeta: 0.5408
Ep 71: 1009s - loss: 0.0026 - f2beta_loss: -4.5604e-01 - fbeta: 0.5816 - val_loss: 0.0037 - val_f2beta_loss: -4.2996e-01 - val_fbeta: 0.5417
Ep 72: 1014s - loss: 0.0025 - f2beta_loss: -4.5733e-01 - fbeta: 0.5834 - val_loss: 0.0037 - val_f2beta_loss: -4.1043e-01 - val_fbeta: 0.5103
Ep 73: 1023s - loss: 0.0025 - f2beta_loss: -4.5824e-01 - fbeta: 0.5841 - val_loss: 0.0037 - val_f2beta_loss: -4.2234e-01 - val_fbeta: 0.5285
Ep 74: 1016s - loss: 0.0025 - f2beta_loss: -4.6047e-01 - fbeta: 0.5893 - val_loss: 0.0037 - val_f2beta_loss: -4.3833e-01 - val_fbeta: 0.5483
Ep 75: 1021s - loss: 0.0025 - f2beta_loss: -4.6171e-01 - fbeta: 0.5904 - val_loss: 0.0036 - val_f2beta_loss: -4.3001e-01 - val_fbeta: 0.5400
Ep 76: 1023s - loss: 0.0025 - f2beta_loss: -4.6138e-01 - fbeta: 0.5906 - val_loss: 0.0036 - val_f2beta_loss: -4.3338e-01 - val_fbeta: 0.5398
Ep 77: 1024s - loss: 0.0025 - f2beta_loss: -4.6238e-01 - fbeta: 0.5908 - val_loss: 0.0037 - val_f2beta_loss: -4.3143e-01 - val_fbeta: 0.5332
Ep 78: 1029s - loss: 0.0025 - f2beta_loss: -4.6362e-01 - fbeta: 0.5933 - val_loss: 0.0036 - val_f2beta_loss: -4.3644e-01 - val_fbeta: 0.5485
Ep 79: 1033s - loss: 0.0025 - f2beta_loss: -4.6336e-01 - fbeta: 0.5942 - val_loss: 0.0037 - val_f2beta_loss: -4.2517e-01 - val_fbeta: 0.5276
Ep 80: 1025s - loss: 0.0025 - f2beta_loss: -4.6531e-01 - fbeta: 0.5948 - val_loss: 0.0036 - val_f2beta_loss: -4.3748e-01 - val_fbeta: 0.5499
Ep 81: 1023s - loss: 0.0025 - f2beta_loss: -4.6557e-01 - fbeta: 0.5963 - val_loss: 0.0036 - val_f2beta_loss: -4.2290e-01 - val_fbeta: 0.5292
Ep 82: 1033s - loss: 0.0025 - f2beta_loss: -4.6548e-01 - fbeta: 0.5956 - val_loss: 0.0035 - val_f2beta_loss: -4.4144e-01 - val_fbeta: 0.5551
Ep 83:1036s - loss: 0.0025 - f2beta_loss: -4.6746e-01 - fbeta: 0.6002 - val_loss: 0.0035 - val_f2beta_loss: -4.3100e-01 - val_fbeta: 0.5417
Ep 84:1070s - loss: 0.0025 - f2beta_loss: -4.6835e-01 - fbeta: 0.5990 - val_loss: 0.0035 - val_f2beta_loss: -4.4616e-01 - val_fbeta: 0.5607
Ep 85:1065s - loss: 0.0025 - f2beta_loss: -4.6986e-01 - fbeta: 0.6031 - val_loss: 0.0036 - val_f2beta_loss: -4.4067e-01 - val_fbeta: 0.5520
Ep 86:1057s - loss: 0.0025 - f2beta_loss: -4.7005e-01 - fbeta: 0.6018 - val_loss: 0.0036 - val_f2beta_loss: -4.5275e-01 - val_fbeta: 0.5739
Ep 87:1040s - loss: 0.0024 - f2beta_loss: -4.7099e-01 - fbeta: 0.6050 - val_loss: 0.0035 - val_f2beta_loss: -4.5337e-01 - val_fbeta: 0.5702
Ep 88:1042s - loss: 0.0025 - f2beta_loss: -4.7009e-01 - fbeta: 0.6030 - val_loss: 0.0037 - val_f2beta_loss: -4.3768e-01 - val_fbeta: 0.5443
Ep 89:1066s - loss: 0.0025 - f2beta_loss: -4.7164e-01 - fbeta: 0.6057 - val_loss: 0.0036 - val_f2beta_loss: -4.3633e-01 - val_fbeta: 0.5499
Ep 90:1079s - loss: 0.0024 - f2beta_loss: -4.7196e-01 - fbeta: 0.6065 - val_loss: 0.0035 - val_f2beta_loss: -4.5498e-01 - val_fbeta: 0.5738 
Ep 91:1073s - loss: 0.0024 - f2beta_loss: -4.7411e-01 - fbeta: 0.6095 - val_loss: 0.0036 - val_f2beta_loss: -4.3319e-01 - val_fbeta: 0.5362 
Ep 92:1052s - loss: 0.0024 - f2beta_loss: -4.7329e-01 - fbeta: 0.6074 - val_loss: 0.0035 - val_f2beta_loss: -4.6061e-01 - val_fbeta: 0.5833
Ep 93:1071s - loss: 0.0024 - f2beta_loss: -4.7734e-01 - fbeta: 0.6150 - val_loss: 0.0035 - val_f2beta_loss: -4.3432e-01 - val_fbeta: 0.5450 
Ep 94:1081s - loss: 0.0024 - f2beta_loss: -4.7523e-01 - fbeta: 0.6118 - val_loss: 0.0035 - val_f2beta_loss: -4.4917e-01 - val_fbeta: 0.5635
Ep 95:1129s - loss: 0.0024 - f2beta_loss: -4.8080e-01 - fbeta: 0.6192 - val_loss: 0.0035 - val_f2beta_loss: -4.4938e-01 - val_fbeta: 0.5637
Ep 96:1146s - loss: 0.0024 - f2beta_loss: -4.7946e-01 - fbeta: 0.6161 - val_loss: 0.0034 - val_f2beta_loss: -4.5790e-01 - val_fbeta: 0.5794
Ep 97:1166s - loss: 0.0024 - f2beta_loss: -4.7959e-01 - fbeta: 0.6168 - val_loss: 0.0035 - val_f2beta_loss: -4.4127e-01 - val_fbeta: 0.5519
Ep 98:1139s - loss: 0.0024 - f2beta_loss: -4.8023e-01 - fbeta: 0.6185 - val_loss: 0.0035 - val_f2beta_loss: -4.6068e-01 - val_fbeta: 0.5758
Ep 99:1161s - loss: 0.0024 - f2beta_loss: -4.7958e-01 - fbeta: 0.6178 - val_loss: 0.0034 - val_f2beta_loss: -4.5531e-01 - val_fbeta: 0.5703
Ep 100:1136s - loss: 0.0024 - f2beta_loss: -4.8133e-01 - fbeta: 0.6202 - val_loss: 0.0034 - val_f2beta_loss: -4.6334e-01 - val_fbeta: 0.5889
Ep 101:1199s - loss: 0.0024 - f2beta_loss: -4.8094e-01 - fbeta: 0.6187 - val_loss: 0.0034 - val_f2beta_loss: -4.4940e-01 - val_fbeta: 0.5687
Ep 102:1162s - loss: 0.0024 - f2beta_loss: -4.8187e-01 - fbeta: 0.6207 - val_loss: 0.0035 - val_f2beta_loss: -4.5118e-01 - val_fbeta: 0.5714
Ep 103:1185s - loss: 0.0024 - f2beta_loss: -4.8355e-01 - fbeta: 0.6239 - val_loss: 0.0034 - val_f2beta_loss: -4.5946e-01 - val_fbeta: 0.5738 
Ep 104:1217s - loss: 0.0024 - f2beta_loss: -4.8263e-01 - fbeta: 0.6221 - val_loss: 0.0034 - val_f2beta_loss: -4.6883e-01 - val_fbeta: 0.5903
Ep 105:933s - loss: 0.0024 - f2beta_loss: -4.8501e-01 - fbeta: 0.6253 - val_loss: 0.0034 - val_f2beta_loss: -4.4871e-01 - val_fbeta: 0.5650
Ep 106:916s - loss: 0.0024 - f2beta_loss: -4.8443e-01 - fbeta: 0.6241 - val_loss: 0.0034 - val_f2beta_loss: -4.5635e-01 - val_fbeta: 0.5756
Ep 107:919s - loss: 0.0024 - f2beta_loss: -4.8336e-01 - fbeta: 0.6219 - val_loss: 0.0035 - val_f2beta_loss: -4.5701e-01 - val_fbeta: 0.5756
Ep 108:913s - loss: 0.0024 - f2beta_loss: -4.8254e-01 - fbeta: 0.6223 - val_loss: 0.0034 - val_f2beta_loss: -4.4970e-01 - val_fbeta: 0.5617
Ep 109:928s - loss: 0.0024 - f2beta_loss: -4.8490e-01 - fbeta: 0.6244 - val_loss: 0.0034 - val_f2beta_loss: -4.5712e-01 - val_fbeta: 0.5764
Ep 110:934s - loss: 0.0024 - f2beta_loss: -4.8526e-01 - fbeta: 0.6247 - val_loss: 0.0035 - val_f2beta_loss: -4.4691e-01 - val_fbeta: 0.5549
Ep 111:931s - loss: 0.0024 - f2beta_loss: -4.8514e-01 - fbeta: 0.6235 - val_loss: 0.0035 - val_f2beta_loss: -4.3598e-01 - val_fbeta: 0.5408
Ep 112:919s - loss: 0.0024 - f2beta_loss: -4.8477e-01 - fbeta: 0.6241 - val_loss: 0.0034 - val_f2beta_loss: -4.5472e-01 - val_fbeta: 0.5700
Ep 113:920s - loss: 0.0024 - f2beta_loss: -4.8547e-01 - fbeta: 0.6249 - val_loss: 0.0034 - val_f2beta_loss: -4.5403e-01 - val_fbeta: 0.5698
Ep 114:920s - loss: 0.0023 - f2beta_loss: -4.8679e-01 - fbeta: 0.6277 - val_loss: 0.0034 - val_f2beta_loss: -4.6622e-01 - val_fbeta: 0.5866
Ep 115:920s - loss: 0.0023 - f2beta_loss: -4.8989e-01 - fbeta: 0.6312 - val_loss: 0.0034 - val_f2beta_loss: -4.5800e-01 - val_fbeta: 0.5744
Ep 116:921s - loss: 0.0023 - f2beta_loss: -4.8997e-01 - fbeta: 0.6320 - val_loss: 0.0033 - val_f2beta_loss: -4.6635e-01 - val_fbeta: 0.5940
Ep 117:917s - loss: 0.0023 - f2beta_loss: -4.8940e-01 - fbeta: 0.6312 - val_loss: 0.0034 - val_f2beta_loss: -4.6420e-01 - val_fbeta: 0.5839
Ep 118:916s - loss: 0.0023 - f2beta_loss: -4.8970e-01 - fbeta: 0.6312 - val_loss: 0.0034 - val_f2beta_loss: -4.6542e-01 - val_fbeta: 0.5843
Ep 119:917s - loss: 0.0023 - f2beta_loss: -4.9085e-01 - fbeta: 0.6327 - val_loss: 0.0034 - val_f2beta_loss: -4.6204e-01 - val_fbeta: 0.5768
Ep 146:956s - loss: 0.0022 - f2beta_loss: -5.1095e-01 - fbeta: 0.6626 - val_loss: 0.0032 - val_f2beta_loss: -4.9182e-01 - val_fbeta: 0.6220
Ep 150:960s - loss: 0.0022 - f2beta_loss: -5.1260e-01 - fbeta: 0.6650 - val_loss: 0.0032 - val_f2beta_loss: -4.9713e-01 - val_fbeta: 0.6268
Ep 152:965s - loss: 0.0022 - f2beta_loss: -5.1435e-01 - fbeta: 0.6673 - val_loss: 0.0032 - val_f2beta_loss: -4.9791e-01 - val_fbeta: 0.6332
Ep 161:977s - loss: 0.0022 - f2beta_loss: -5.1655e-01 - fbeta: 0.6706 - val_loss: 0.0031 - val_f2beta_loss: -5.0185e-01 - val_fbeta: 0.6350 
Ep 167:990s - loss: 0.0021 - f2beta_loss: -5.1847e-01 - fbeta: 0.6741 - val_loss: 0.0031 - val_f2beta_loss: -5.0431e-01 - val_fbeta: 0.6385
Ep 170:1011s - loss: 0.0021 - f2beta_loss: -5.1986e-01 - fbeta: 0.6748 - val_loss: 0.0031 - val_f2beta_loss: -5.0438e-01 - val_fbeta: 0.6444
Ep 186:1075s - loss: 0.0021 - f2beta_loss: -5.2981e-01 - fbeta: 0.6898 - val_loss: 0.0030 - val_f2beta_loss: -5.1324e-01 - val_fbeta: 0.6475
Ep 189:1048s - loss: 0.0021 - f2beta_loss: -5.2966e-01 - fbeta: 0.6894 - val_loss: 0.0030 - val_f2beta_loss: -5.1321e-01 - val_fbeta: 0.6507
Ep 190:1069s - loss: 0.0021 - f2beta_loss: -5.2984e-01 - fbeta: 0.6894 - val_loss: 0.0030 - val_f2beta_loss: -5.1409e-01 - val_fbeta: 0.6511
Ep 206:1378s - loss: 0.0021 - f2beta_loss: -5.3314e-01 - fbeta: 0.6938 - val_loss: 0.0030 - val_f2beta_loss: -5.1926e-01 - val_fbeta: 0.6610 - best
Ep 221:1412s - loss: 0.0020 - f2beta_loss: -5.3975e-01 - fbeta: 0.7030 - val_loss: 0.0030 - val_f2beta_loss: -5.1899e-01 - val_fbeta: 0.6539

Augmentations added:
Ep 245: 936s - loss: 0.0027 - f2beta_loss: -4.2697e-01 - fbeta: 0.5140 - val_loss: 0.0033 - val_f2beta_loss: -4.6704e-01 - val_fbeta: 0.5875 F2: 0.332028 THR: 0.5
Ep 250: 906s - loss: 0.0024 - f2beta_loss: -4.7037e-01 - fbeta: 0.5987 - val_loss: 0.0033 - val_f2beta_loss: -4.7339e-01 - val_fbeta: 0.5995 F2: 0.336537 THR: 0.7
Ep 271: 924s - loss: 0.0024 - f2beta_loss: -4.8413e-01 - fbeta: 0.6203 - val_loss: 0.0031 - val_f2beta_loss: -4.9390e-01 - val_fbeta: 0.6254 F2: 0.325015 THR: 0.5
Ep 272: 902s - loss: 0.0024 - f2beta_loss: -4.7992e-01 - fbeta: 0.6131 - val_loss: 0.0032 - val_f2beta_loss: -4.8649e-01 - val_fbeta: 0.6086 F2: 0.339348 THR: 0.5
Ep 278: 901s - loss: 0.0024 - f2beta_loss: -4.8172e-01 - fbeta: 0.6161 - val_loss: 0.0032 - val_f2beta_loss: -4.8239e-01 - val_fbeta: 0.6066 F2: 0.330009 THR: 0.5
Ep 279: 903s - loss: 0.0024 - f2beta_loss: -4.8130e-01 - fbeta: 0.6157 - val_loss: 0.0031 - val_f2beta_loss: -4.9303e-01 - val_fbeta: 0.6212 F2: 0.331992 THR: 0.5
Ep 282: 905s - loss: 0.0024 - f2beta_loss: -4.8246e-01 - fbeta: 0.6169 - val_loss: 0.0031 - val_f2beta_loss: -4.9521e-01 - val_fbeta: 0.6257 F2: 0.334868 THR: 0.5
Ep 285: 897s - loss: 0.0024 - f2beta_loss: -4.8273e-01 - fbeta: 0.6193 - val_loss: 0.0032 - val_f2beta_loss: -4.9384e-01 - val_fbeta: 0.6273 F2: 0.326069 THR: 0.6
Ep 288: 895s - loss: 0.0023 - f2beta_loss: -4.8757e-01 - fbeta: 0.6244 - val_loss: 0.0031 - val_f2beta_loss: -5.0244e-01 - val_fbeta: 0.6335 F2: 0.344849 THR: 0.6
Ep 296: 896s - loss: 0.0023 - f2beta_loss: -4.8983e-01 - fbeta: 0.6286 - val_loss: 0.0031 - val_f2beta_loss: -5.0429e-01 - val_fbeta: 0.6386 F2: 0.334089 THR: 0.7
Ep 314: 920s - loss: 0.0022 - f2beta_loss: -5.0233e-01 - fbeta: 0.6475 - val_loss: 0.0030 - val_f2beta_loss: -5.1504e-01 - val_fbeta: 0.6522 F2: 0.339238 THR: 0.5
Ep 316: 924s - loss: 0.0022 - f2beta_loss: -5.0310e-01 - fbeta: 0.6470 - val_loss: 0.0030 - val_f2beta_loss: -5.0680e-01 - val_fbeta: 0.6407 F2: 0.334560 THR: 0.7
Ep 324: 930s - loss: 0.0022 - f2beta_loss: -5.0635e-01 - fbeta: 0.6530 - val_loss: 0.0030 - val_f2beta_loss: -5.1703e-01 - val_fbeta: 0.6523 F2: 0.345363 THR: 0.6
Ep 331:1032s - loss: 0.0022 - f2beta_loss: -5.0900e-01 - fbeta: 0.6574 - val_loss: 0.0030 - val_f2beta_loss: -5.1841e-01 - val_fbeta: 0.6527 F2: 0.346996 THR: 0.6
Ep 349: 979s - loss: 0.0022 - f2beta_loss: -5.1238e-01 - fbeta: 0.6625 - val_loss: 0.0030 - val_f2beta_loss: -5.2118e-01 - val_fbeta: 0.6562 F2: 0.352160 THR: 0.6
Ep 360:1002s - loss: 0.0022 - f2beta_loss: -5.1598e-01 - fbeta: 0.6664 - val_loss: 0.0029 - val_f2beta_loss: -5.2450e-01 - val_fbeta: 0.6636 F2: 0.342303 THR: 0.7
Ep 388:1212s - loss: 0.0021 - f2beta_loss: -5.2196e-01 - fbeta: 0.6744 - val_loss: 0.0029 - val_f2beta_loss: -5.2924e-01 - val_fbeta: 0.6684 F2: 0.345518 THR: 0.5
Ep 399:1314s - loss: 0.0021 - f2beta_loss: -5.2379e-01 - fbeta: 0.6782 - val_loss: 0.0029 - val_f2beta_loss: -5.3248e-01 - val_fbeta: 0.6712 F2: 0.345103 THR: 0.6
Ep 400:1334s - loss: 0.0021 - f2beta_loss: -5.2267e-01 - fbeta: 0.6755 - val_loss: 0.0029 - val_f2beta_loss: -5.2986e-01 - val_fbeta: 0.6676 F2: 0.347600 THR: 0.6
Ep 415: 965s - loss: 0.0022 - f2beta_loss: -5.0252e-01 - fbeta: 0.6469 - val_loss: 0.0030 - val_f2beta_loss: -5.1127e-01 - val_fbeta: 0.6403 F2: 0.346221 THR: 0.6
Ep 442: 926s - loss: 0.0022 - f2beta_loss: -5.1347e-01 - fbeta: 0.6635 - val_loss: 0.0029 - val_f2beta_loss: -5.1708e-01 - val_fbeta: 0.6529 F2: 0.342543 THR: 0.6
Ep 447: 931s - loss: 0.0022 - f2beta_loss: -5.1648e-01 - fbeta: 0.6670 - val_loss: 0.0029 - val_f2beta_loss: -5.2051e-01 - val_fbeta: 0.6553 F2: 0.347902 THR: 0.6
Ep 448: 935s - loss: 0.0022 - f2beta_loss: -5.1654e-01 - fbeta: 0.6674 - val_loss: 0.0030 - val_f2beta_loss: -5.2718e-01 - val_fbeta: 0.6648 F2: 0.351656 THR: 0.5
Ep 469:1240s - loss: 0.0021 - f2beta_loss: -5.2097e-01 - fbeta: 0.6740 - val_loss: 0.0029 - val_f2beta_loss: -5.3055e-01 - val_fbeta: 0.6704 F2: 0.342368 THR: 0.5
Ep 474:1272s - loss: 0.0021 - f2beta_loss: -5.2440e-01 - fbeta: 0.6797 - val_loss: 0.0029 - val_f2beta_loss: -5.2612e-01 - val_fbeta: 0.6602 F2: 0.351992 THR: 0.7
Ep 484:1219s - loss: 0.0021 - f2beta_loss: -5.2583e-01 - fbeta: 0.6805 - val_loss: 0.0029 - val_f2beta_loss: -5.3712e-01 - val_fbeta: 0.6760 F2: 0.348851 THR: 0.7
Ep 488:1236s - loss: 0.0021 - f2beta_loss: -5.2747e-01 - fbeta: 0.6829 - val_loss: 0.0029 - val_f2beta_loss: -5.3363e-01 - val_fbeta: 0.6713 F2: 0.352756 THR: 0.6
Ep 499: 888s - loss: 0.0021 - f2beta_loss: -5.2679e-01 - fbeta: 0.6817 - val_loss: 0.0029 - val_f2beta_loss: -5.3147e-01 - val_fbeta: 0.6687 F2: 0.350053 THR: 0.6
Ep 523: 919s - loss: 0.0021 - f2beta_loss: -5.3010e-01 - fbeta: 0.6864 - val_loss: 0.0028 - val_f2beta_loss: -5.3569e-01 - val_fbeta: 0.6742 F2: 0.350012 THR: 0.6
Ep 527: 926s - loss: 0.0021 - f2beta_loss: -5.3355e-01 - fbeta: 0.6927 - val_loss: 0.0028 - val_f2beta_loss: -5.3870e-01 - val_fbeta: 0.6778 F2: 0.350323 THR: 0.6
Ep 528: 931s - loss: 0.0020 - f2beta_loss: -5.3376e-01 - fbeta: 0.6925 - val_loss: 0.0028 - val_f2beta_loss: -5.3790e-01 - val_fbeta: 0.6770 F2: 0.352631 THR: 0.6
Ep 539: 950s - loss: 0.0021 - f2beta_loss: -5.3606e-01 - fbeta: 0.6945 - val_loss: 0.0028 - val_f2beta_loss: -5.4146e-01 - val_fbeta: 0.6818 F2: 0.350123 THR: 0.7
Ep 564: 894s - loss: 0.0020 - f2beta_loss: -5.3511e-01 - fbeta: 0.6942 - val_loss: 0.0028 - val_f2beta_loss: -5.4063e-01 - val_fbeta: 0.6796 F2: 0.352337 THR: 0.7
Ep 581: 952s - loss: 0.0020 - f2beta_loss: -5.3747e-01 - fbeta: 0.6971 - val_loss: 0.0028 - val_f2beta_loss: -5.4162e-01 - val_fbeta: 0.6821 F2: 0.352358 THR: 0.7

'''
