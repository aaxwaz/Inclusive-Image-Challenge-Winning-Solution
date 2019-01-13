from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from a00_common_functions import *
from a01_neural_nets import *
#from neural_nets.a00_augmentation_functions import *
from a02_common_training_structures import *
#from albumentations import *
import random
from tensorflow import flags
from multilabel_data_generator_ext_npzImage import * 
from keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input
import os 

FLAGS = flags.FLAGS

if __name__ == '__main__':

    flags.DEFINE_integer("model_index", 4, 
                        "Model index")

    flags.DEFINE_bool("train_new_model", False,
                      "Whether to train model from scratch")

    print("Using GPU device: ", (0,1))
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    MODELS_PATH = MODELS_PATH.split('/')
    MODELS_PATH[-2] = 'model_' + str(FLAGS.model_index)
    MODELS_PATH = '/'.join(MODELS_PATH)

    if not os.path.isdir(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    HISTORY_FOLDER_PATH = MODELS_PATH + "history/"

    print("Path for this training is: ", MODELS_PATH)


def process_single_item_npz(inp_file, data_path=None, box_size=299, name_ext='.jpg'):

    if data_path:
        f = data_path + inp_file + name_ext
    else:
        f = DATASET_PATH + inp_file + name_ext

    try:
        im_full_big = load_img_npz(f)
    except:
        im_full_big = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        print("Unable to load this image: ", f)

    if im_full_big is None:
        im_full_big = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    return im_full_big


def get_target(batch_files, image_classes):
    target = np.zeros((len(batch_files), 7178), dtype=np.float32)
    for i, b in enumerate(batch_files):
        for el in image_classes[b]:
            target[i, el] = image_classes[b][el]
    return target


def batch_generator_val(files, image_classes, batch_size, data_path=None):
    nn_shape = 299
    threads = 8

    # threads = 1
    p = ThreadPool(threads)
    process_item_func = partial(process_single_item_npz, data_path=data_path, box_size=nn_shape, name_ext='.npz')
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

        batch_images = xception_preprocess_input(batch_images)
        yield batch_images, batch_target

def prepare_val_images(val_images):
    train_path = DATASET_PATH+'train/'
    for img in val_images:
        if os.path.exists(train_path+img+'.jpg'):
            shutil.move(train_path+img+'.jpg', VAL_DATAPATH+img+'.jpg')
    
    num_val, num_train = len(os.listdir(VAL_DATAPATH)), len(os.listdir(train_path))

    print("\ndone val images preparation. Numbers of train and val images are: {} {} \n".format(num_train, num_val))

def train_single_model(train_files, valid_files):
    import keras.backend as K
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from keras.optimizers import Adam, SGD
    from keras.losses import mean_squared_error
    from keras.models import load_model
    import tensorflow as tf
    from keras.utils import multi_gpu_model

    image_classes = get_classes_for_images_dict()

    tuning_test_images, tuning_test_labels = get_tuning_labels_data_for_validation(299)

    gen = ImageDataGenerator(horizontal_flip = True,
                             vertical_flip = True,
                             width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             channel_shift_range=0.1,
                             shear_range = 0.1,
                             zoom_range = 0.1, 
                             rotation_range = 10, 
                             preprocessing_function=xception_preprocess_input)

    #restore = 1
    patience = 50
    epochs = 1000
    optim_type = 'Adam'
    learning_rate = 0.001
    cnn_type = 'new_xception'
    #num_gpus = 2
    print('Creating and compiling {}...'.format(cnn_type))
    print("Using {} gpus. ".format(2))
    val_data_path = '/home/ec2-user/.inclusive/train/val_images/'
    print("Num of val images: ", len(os.listdir(val_data_path)), '\n')

    ##### Prepare the model #####
    #model = get_model_resnet50_336()
    #model = get_model_densenet()
    model = get_model_xception()
    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, fbeta])
    ##### Prepare the model #####

    cache_model_path = MODELS_PATH + '{}_temp.h5'.format(cnn_type)
    final_model_path = MODELS_PATH + '{}.h5'.format(cnn_type)
    if os.path.isfile(final_model_path):
        print('Model already exists {}.'.format(final_model_path))

    if FLAGS.train_new_model:
        if os.path.isdir(MODELS_PATH):
            print("Removing previous models if any.  ")
            shutil.rmtree(MODELS_PATH)
     
    if not os.path.isdir(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    if not os.path.isdir(HISTORY_FOLDER_PATH):
        os.mkdir(HISTORY_FOLDER_PATH)
  
    if not FLAGS.train_new_model:
        if os.path.isfile(final_model_path):
            print('Load model from last point: ', final_model_path)
            #model = load_model(final_model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
            model.load_weights(final_model_path)
        else:
            print("\nCouldn't find previously trained models. Exit. ")
            return 0
    else:
        print("\nStarted to train new model. \n")

    np.random.seed(10)
    valid_files = np.random.choice(valid_files, 4000)
    print(valid_files[:4])

    print('Fitting model...')
    batch_size = 48
    print('Batch size: {}'.format(batch_size))
    steps_per_epoch = 96000 // batch_size
    validation_steps = len(valid_files) // (batch_size)
    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    gen_flow = gen.flow_from_directory(DATASET_PATH, target_size=(299, 299), batch_size=batch_size, class_mode='multilabel', multilabel_classes=image_classes, n_class=7178)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
        MyModelCheckpoint(cache_model_path[:-7]+'latest.h5', monitor='val_fbeta', mode='max', save_weights_only=True, save_best_only=True, verbose=0),
        #MyModelCheckpoint(final_model_path, monitor='val_fbeta', mode='max', save_weights_only=True, save_best_only=True, verbose=1),
        # ModelCheckpoint(cache_model_path[:-3] + '_{epoch:02d}.h5', monitor='val_fbeta', mode='max', verbose=0),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_{}_lr_{}_optim_{}.csv'.format(cnn_type,
                                                                               learning_rate,
                                                                               optim_type), append=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=1e-9, min_delta=0.00001, verbose=0, mode='min'),
        # CyclicLR(base_lr=0.0001, max_lr=0.001, step_size=1000)
        ModelCheckpoint_F2Score(cache_model_path[:-3] + '_byTestScore_{epoch:02d}.h5', save_best_only=True, save_weights_only=True, 
                                mode='max', patience=patience, verbose=0,
                                validation_data=(tuning_test_images, tuning_test_labels)),
    ]
    history = model.fit_generator(generator=gen_flow,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator_val(np.array(list(valid_files)), image_classes, batch_size, val_data_path),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=10,
                                  initial_epoch=0,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss: {} [Ep: {}]'.format(min_loss, len(history.history['val_loss'])))
    #model.load_weights(cache_model_path)
    #model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    save_history_figure(history, filename[:-4] + '.png')
    # del model
    # K.clear_session()
    return min_loss

def create_models_new_xception():
    train_files, valid_files = get_train_valid_split_of_files()
    print('Split files train:', len(train_files))
    print('Split files valid:', len(valid_files))
    prepare_val_images(valid_files)
    train_single_model(train_files, valid_files)

if __name__ == '__main__':
    start_time = time.time()
    create_models_new_xception()
    print('Time: {:.0f} sec'.format(time.time() - start_time))

