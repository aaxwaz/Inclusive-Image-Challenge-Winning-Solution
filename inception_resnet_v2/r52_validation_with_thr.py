# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a02_common_training_structures import *
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


PRECISION = 6
EPS = 0.0001


def get_tuning_matrix_from_cache_tst(folder_path, cache_file):
    if not os.path.isfile(cache_file) or 0:
        test_image_classes = get_classes_for_tst_images_dict()
        print('Test valid:', len(test_image_classes))
        matrix_pred = np.zeros((len(test_image_classes), 7178), dtype=np.float32)
        matrix_real = np.zeros((len(test_image_classes), 7178), dtype=np.float32)
        for i, id in enumerate(sorted(list(test_image_classes.keys()))):
            print('Go for {} {}'.format(i, id))
            cache_path = folder_path + id + '.pkl'
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


def get_full_matrix_from_cache_tst(folder_path, cache_file):
    if not os.path.isfile(cache_file) or 0:
        test_files = glob.glob(INPUT_PATH + 'stage_1_test_images/*.jpg')
        print('Test valid:', len(test_files))
        matrix_pred = np.zeros((len(test_files), 7178), dtype=np.float32)
        for i, f in enumerate(sorted(list(test_files))):
            id = os.path.basename(f)[:-4]
            print('Go for {} {}'.format(i, id))
            cache_path = folder_path + id + '.pkl'
            preds = load_from_file(cache_path)
            preds = preds.mean(axis=0)
            total = 0
            for j in range(len(preds)):
                if preds[j] >= EPS:
                    matrix_pred[i, j] = preds[j]
                    total += 1
            print('Stored: {}'.format(total))
        save_in_file(matrix_pred, cache_file)
    else:
        matrix_pred = load_from_file(cache_file)
    return matrix_pred


def get_score_multithread(matrix_pred, matrix_real, thr_arr_orig, class_id, thr):
    thr_arr = thr_arr_orig.copy()
    thr_arr[class_id] = thr
    m = matrix_pred.copy()
    m[m > thr_arr] = 1
    m[m <= thr_arr] = 0
    score = fbeta_score(matrix_real, m, beta=2, average='samples')
    return score, thr


def find_optimal_score(matrix_pred, matrix_real, start_point, end_point, min_number_of_preds, default_score):
    flt = matrix_real.sum(axis=0)
    thr_arr = np.zeros_like(flt)

    p = ThreadPool(7)

    thr_arr[...] = default_score
    thr_arr[flt > 0] = 0.5
    best_score = 0
    for zz in range(2):
        for i in range(len(thr_arr)):
            if flt[i] >= min_number_of_preds:
                test_points = list(np.linspace(start_point, end_point, 99))
                res = p.starmap(get_score_multithread, [[matrix_pred, matrix_real, thr_arr, i, thr] for thr in test_points])
                res = sorted(res, key=lambda tup: (tup[0], tup[1]), reverse=True)
                best_score = res[0][0]
                best_thr = res[0][1]
                thr_arr[i] = best_thr
                print('{} {} {:.3f} {:.6f}'.format(i, flt[i], best_thr, best_score))
    print('Predicted score: {:.6f}. Min entries in class: {}'.format(best_score, min_number_of_preds))
    return thr_arr


def create_submission(folder_path, thr_arr, out_file):
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    out = open(out_file, 'w')
    out.write('image_id,labels\n')
    test_files = glob.glob(INPUT_PATH + 'stage_1_test_images/*.jpg')
    print('Test valid:', len(test_files))
    for i, f in enumerate(test_files):
        id = os.path.basename(f)[:-4]
        cache_path = folder_path + id + '.pkl'
        preds = load_from_file(cache_path)
        preds = preds.mean(axis=0)
        total = 0
        out.write(id + ',')
        for j in range(len(preds)):
            if preds[j] >= thr_arr[j]:
                out.write(index_arr_backward[j] + ' ')
                total += 1
        out.write('\n')
        # print('Go for {} {}. Stored: {}'.format(i, id, total))
    out.close()


def create_submission_with_replace_for_empty(folder_path, thr_arr, out_file, limit):
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    out = open(out_file, 'w')
    out.write('image_id,labels\n')
    test_files = glob.glob(INPUT_PATH + 'stage_1_test_images/*.jpg')
    print('Test valid:', len(test_files))
    for i, f in enumerate(test_files):
        id = os.path.basename(f)[:-4]
        cache_path = folder_path + id + '.pkl'
        preds = load_from_file(cache_path)
        preds = preds.mean(axis=0)
        total = 0

        copy_preds = preds.copy()
        preds_index = preds.argsort()[-limit:][::-1]
        copy_preds[copy_preds < thr_arr] = 0

        total = 0
        if copy_preds.sum() > 0:
            out.write(id + ',')
            for j in range(len(preds)):
                if preds[j] >= thr_arr[j]:
                    out.write(index_arr_backward[j] + ' ')
                    total += 1
            out.write('\n')
        else:
            s = str(id)
            out.write(id + ',')
            for p in preds_index:
                out.write(index_arr_backward[p] + ' ')
                s += str(p) + ' ' + str(index_arr_backward[p]) + ' '
                total += 1
            out.write('\n')
            print(s)
            total += 1
        # print('Go for {} {}. Stored: {}'.format(i, id, total))
        print('Total fixed: {}'.format(total))

    out.close()


if __name__ == '__main__':
    start_time = time.time()
    if 1:
        start_point = 0.01
        end_point = 0.99
        min_number_of_entries = 1
        default_value = 0.99
        matrix_pred, matrix_real = get_tuning_matrix_from_cache_tst(OUTPUT_PATH + 'cache_inception_resnet_v2/',
                                                         OUTPUT_PATH + 'cache_inception_resnet_v2_test.pklz')
        thr_arr = find_optimal_score(matrix_pred, matrix_real, start_point, end_point, min_number_of_entries, default_value)
        save_in_file(thr_arr, OUTPUT_PATH + 'thr_arr_inception_resnet_v2_sp_{}_ep_{}_min_{}_def_{}.pklz'.
                     format(start_point, end_point, min_number_of_entries, default_value))
        matrix_pred = create_submission(OUTPUT_PATH + 'cache_inception_resnet_v2_test/', thr_arr, SUBM_PATH + 'optim_thr_resnet_v2.csv')


'''
v1: inception_resnet_v2_temp_183.h5
[THR: 0.01 - 0.99, Min entries: 1] LS: 0.629954 LB: 0.503
[THR: 0.001 - 0.999, Min entries: 1] LS: 0.639413 LB: 0.507
[THR: 0.0001 - 0.9999, Min entries: 1] LS: 0.638779
[THR: 0.1 - 0.9, Min entries: 3, Default: 0.9] LS: 0.480826 LB: 0.435


v2: inception_resnet_v2_temp_320.h5
[THR: 0.001 - 0.999, Min entries: 1] LS: 0.642779 LB: 0.511
Limit on 7 classes LB: 0.510
Replace empty with 4 most probable: 0.514
'''