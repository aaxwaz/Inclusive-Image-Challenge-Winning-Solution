from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from a00_common_functions import *
from a01_neural_nets import *
#from neural_nets.a00_augmentation_functions import *
from a02_common_training_structures import *
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.xception import preprocess_input as xception_preprocess_input

from tensorflow import flags

import gc 

FLAGS = flags.FLAGS

PRECISION = 6
EPS = 0.0001

if __name__ == '__main__':
    
    CACHE_PATH_TEST = OUTPUT_PATH + 'cache_inception_resnet_weimin_0/'

if not os.path.isdir(CACHE_PATH_TEST):
    os.mkdir(CACHE_PATH_TEST)

def get_target(batch_files, image_classes):
    target = np.zeros((len(batch_files), 7178), dtype=np.uint8)
    for i, b in enumerate(batch_files):
        for el in image_classes[b]:
            target[i, el] = 1
    return target

def process_single_item(inp_file, box_size, dataset_path=None):
    if dataset_path:
        f = dataset_path +  inp_file + '.npz'
    else:
        f = DATASET_PATH + 'train/' + inp_file + '.npz'

    im_part = read_single_image(f)

    if im_part is None:
        im_part = np.zeros((box_size, box_size, 3), dtype=np.uint8)

    im_part = cv2.resize(im_part, (box_size, box_size), cv2.INTER_LANCZOS4)

    return im_part

def process_single_item_npz(inp_file, box_size, dataset_path=None):
    if dataset_path:
        f = dataset_path +  inp_file + '.npz'
    else:
        f = DATASET_PATH + 'train/' + inp_file + '.npz'

    im_part = load_from_file_fast(f)

    return im_part

PRECISION = 6
EPS = 0.0001

OUTPUT_DIM = 7178

FILE_EXT = '.pkl'

def get_tuning_matrix_from_cache_tst(folder_path, cache_file):
    if not os.path.isfile(cache_file) or 0:
        if OUTPUT_DIM == 7178:
            test_image_classes = get_classes_for_tst_images_dict()
        else:
            test_image_classes = get_classes_for_tst_images_dict_top_491()
        print('Test valid:', len(test_image_classes))
        matrix_pred = np.zeros((len(test_image_classes), OUTPUT_DIM), dtype=np.float32)
        matrix_real = np.zeros((len(test_image_classes), OUTPUT_DIM), dtype=np.float32)
        for i, id in enumerate(sorted(list(test_image_classes.keys()))):
            print('Go for {} {}'.format(i, id))
            cache_path = folder_path + id + FILE_EXT
            preds = load_from_file(cache_path)
            preds = preds.mean(axis=0)
            total = 0
            for j in range(len(preds)):
                if preds[j] >= EPS:
                    matrix_pred[i, j] = preds[j]
                    total += 1
            for el in test_image_classes[id]:
                matrix_real[i, el] = 1
            print('Stored: {}'.format(total))
        save_in_file((matrix_pred, matrix_real), cache_file)
    else:
        matrix_pred, matrix_real = load_from_file(cache_file)
    return matrix_pred, matrix_real


def get_score_multithread(matrix_pred, matrix_real, thr_arr_orig, class_id, thr):
    thr_arr = thr_arr_orig.copy()
    thr_arr[class_id] = thr
    m = matrix_pred.copy()
    m[m > thr_arr] = 1
    m[m <= thr_arr] = 0
    score = fbeta_score(matrix_real, m, beta=2, average='samples')
    return score, thr


def find_optimal_score_fast(matrix_pred, matrix_real, sp=0.1, ep=0.9, minThr=1, default=0.9):
    min_number_of_preds = minThr

    flt = matrix_real.sum(axis=0)
    useful_index = np.where(flt>=min_number_of_preds)[0]

    thr_arr_all = np.ones(len(flt)) * default
    thr_arr_all[useful_index] = 0.5
    thr_arr_useful = np.ones(len(useful_index)) * 0.5

    assert sum(thr_arr_all==0.5) == len(thr_arr_useful)
    print("Number of columns to explore threshold for: ", len(thr_arr_useful))

    matrix_pred_useful = matrix_pred[:, useful_index]
    matrix_real_useful = matrix_real[:, useful_index]

    p = ThreadPool(8)

    test_points = list(np.linspace(sp, ep, int(ep/sp)))
    print("Exploring space for ", test_points)

    #thr_arr[...] = 0.9999
    #thr_arr[flt > 0] = 0.5
    best_score = 0
    for zz in range(2):
        count_pos = 0 
        for i in range(len(thr_arr_all)):
            if flt[i] >= min_number_of_preds:
                res = p.starmap(get_score_multithread, [[matrix_pred_useful, matrix_real_useful, thr_arr_useful, count_pos, thr] for thr in test_points])
                res = sorted(res, key=lambda tup: (tup[0], tup[1]), reverse=True)
                best_score = res[0][0]
                best_thr = res[0][1]
                thr_arr_useful[count_pos] = best_thr
                thr_arr_all[i] = best_thr

                print('{} {} {:.3f} {:.6f}'.format(i, flt[i], best_thr, best_score))
                count_pos += 1

    ## calculate final score 
    m = matrix_pred.copy()
    m[m > thr_arr_all] = 1.0
    m[m <= thr_arr_all] = 0.0
    final_score = fbeta_score(matrix_real, m, beta=2, average='samples')
    
    print('Predicted score: {:.6f}. Min entries in class: {}'.format(final_score, min_number_of_preds))
    return thr_arr_all

def create_submission(model_path, is_weights = False):
    from keras.models import load_model
    from keras.utils import multi_gpu_model
    from keras.optimizers import Adam, SGD

    print('\nModel weights: ', model_path, '\n')
    if is_weights:
        optim_type = 'Adam'
        learning_rate = 0.001

        ##### Prepare the model #####
        model = get_model_inception_resnet_v2()
        if optim_type == 'SGD':
            optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        else:
            optim = Adam(lr=learning_rate)
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f2beta_loss, fbeta])

        model.load_weights(model_path)
    else:
        model = load_model(model_path, custom_objects={'f2beta_loss': f2beta_loss, 'fbeta': fbeta})
    test_files = glob.glob(TUNING_IMAGE_PATH)
    print('Test valid:', len(test_files))

    for i, f in enumerate(sorted(test_files)):
        if i % 500 == 0:
            print(i)
        id = os.path.basename(f)[:-4]
        path = f

        cache_path = CACHE_PATH_TEST + id + FILE_EXT
        if not os.path.isfile(cache_path):
            im_full_big = load_img_npz(path)

            batch_images = []
            batch_images.append(im_full_big.copy())
            batch_images.append(im_full_big[:, ::-1, :].copy())
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_images = inception_resnet_preprocess_input(batch_images)

            preds = model.predict(batch_images)
            preds[preds < EPS] = 0
            preds = np.round(preds, PRECISION)
            save_in_file(preds, cache_path)


if __name__ == '__main__':
    start_time = time.time()

    print("Using GPU device: ", (0,1))
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

    model_saved_path = ROOT_PATH + 'model_0/inception_resnet_v2_latest.h5'

    create_submission(model_saved_path, is_weights=True)

    matrix_pred, matrix_real = get_tuning_matrix_from_cache_tst(CACHE_PATH_TEST, OUTPUT_PATH + 'cache_inception_resnet_weimin_v0.pklz')

    sp=0.1
    ep=0.9
    minThr=1
    default=0.9

    thr_arr = find_optimal_score_fast(matrix_pred, matrix_real, sp=sp, ep=ep, minThr=minThr, default=default)
    
    save_in_file(thr_arr, OUTPUT_PATH + 'thr_arr_inception_resnet_version_1_sp_{}_ep_{}_min_{}_def_{}.pklz'.format(sp, ep, minThr, default))

    print('Time: {:.0f} sec'.format(time.time() - start_time))

