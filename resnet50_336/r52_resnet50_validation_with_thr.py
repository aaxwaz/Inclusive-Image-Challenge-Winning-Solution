# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a02_common_training_structures import *
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


PRECISION = 6
EPS = 0.01


def get_sparse_matrix_from_cache_train(folder_path, cache_file):

    if not os.path.isfile(cache_file):
        train_files, valid_files = get_train_valid_split_of_files()
        print('Split files valid:', len(valid_files))
        matrix = dok_matrix((len(valid_files), 7178), dtype=np.float32)
        for i, id in enumerate(valid_files):
            print('Go for {} {}'.format(i, id))
            cache_path = folder_path + id + '.pkl'
            preds = load_from_file(cache_path)
            preds = preds.mean(axis=0)
            total = 0
            for j in range(len(preds)):
                if preds[j] >= EPS:
                    matrix[i, j] = preds[j]
                    total += 1
            print('Stored: {}'.format(total))
        save_in_file(matrix, cache_file)
    else:
        matrix = load_from_file(cache_file)
    return matrix


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


def find_optimal_score(matrix_pred, matrix_real, start_point, end_point, min_number_of_entries, default_value):
    flt = matrix_real.sum(axis=0)
    thr_arr = np.zeros_like(flt)

    p = ThreadPool(7)

    min_number_of_preds = min_number_of_entries
    thr_arr[...] = default_value
    thr_arr[flt > 0] = 0.5
    best_score = 0
    for zz in range(2):
        for i in range(len(thr_arr)):
            if flt[i] >= min_number_of_preds:
                test_points = list(np.linspace(start_point, end_point, 100))
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


def draw_distributions(matrix_pred_valid, matrix_real_valid, matrix_pred_test, matrix_real_test, matrix_pred_test_full):
    class_id = 5415
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    dcode, dreal = get_classes_dict()
    print(index_arr_backward[class_id], dcode[index_arr_backward[class_id]])
    print('Validation stat. Mean: {:.6f} Std: {:.6f}'.format(matrix_pred_valid[:, class_id].mean(), matrix_pred_valid[:, class_id].std()))
    print('Test full  stat. Mean: {:.6f} Std: {:.6f}'.format(matrix_pred_test_full[:, class_id].mean(), matrix_pred_test_full[:, class_id].std()))
    print('Test tune  stat. Mean: {:.6f} Std: {:.6f}'.format(matrix_pred_test[:, class_id].mean(),
                                                             matrix_pred_test[:, class_id].std()))

    # Find stat for validation
    preds = matrix_pred_valid.copy()
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    best_thr = -1
    best_score = -1
    check_thrs = list(np.linspace(0.0, 1, 10+1))
    # check_thrs = [0.1]
    for thr in check_thrs:
        p1 = matrix_pred_valid[:, class_id].copy()
        p1[p1 > thr] = 1
        p1[p1 <= thr] = 0
        preds[:, class_id] = p1

        score_valid = fbeta_score(matrix_real_valid, preds, beta=2, average='samples')
        print("Optimal thr search: {:.6f} {:.6f}".format(thr, score_valid))
        if score_valid > best_score:
            best_score = score_valid
            best_thr = thr
    print('Best validation score: {:.6f} for thr: {}'.format(best_score, best_thr))

    labels_true = len(matrix_real_valid[matrix_real_valid[:, class_id] > 0])
    print('True labels in validation: {} out of {}. Fraction: {:.6f}'.format(labels_true, len(matrix_real_valid), labels_true/len(matrix_real_valid)))

    p1 = matrix_pred_valid[:, class_id].copy()
    p1[p1 > 0.5] = 1
    p1[p1 <= 0.5] = 0
    print('True labels in predictions for 0.5 default THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32), p1.mean()))

    p1 = matrix_pred_valid[:, class_id].copy()
    p1[p1 > best_thr] = 1
    p1[p1 <= best_thr] = 0
    print('True labels in predictions for optimal THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32), p1.mean()))

    # Find stat for test
    preds = matrix_pred_test.copy()
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    best_thr = -1
    best_score = -1
    check_thrs = list(np.linspace(0.0, 1, 30 + 1))
    for thr in check_thrs:
        p1 = matrix_pred_test[:, class_id].copy()
        p1[p1 > thr] = 1
        p1[p1 <= thr] = 0
        preds[:, class_id] = p1

        score_valid = fbeta_score(matrix_real_test, preds, beta=2, average='samples')
        print("Optimal thr search: {:.6f} {:.6f}".format(thr, score_valid))
        if score_valid > best_score:
            best_score = score_valid
            best_thr = thr
    print('Best test score: {:.6f} for thr: {}'.format(best_score, best_thr))

    labels_true = len(matrix_real_test[matrix_real_test[:, class_id] > 0])
    print('True labels in test: {} out of {}. Fraction: {:.6f}'.format(labels_true, len(matrix_real_test),
                                                                             labels_true / len(matrix_real_test)))

    p1 = matrix_pred_test[:, class_id].copy()
    p1[p1 > 0.5] = 1
    p1[p1 <= 0.5] = 0
    print('True labels in predictions for 0.5 default THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32), p1.mean()))

    p1 = matrix_pred_test[:, class_id].copy()
    p1[p1 > best_thr] = 1
    p1[p1 <= best_thr] = 0
    print('True labels in predictions for optimal THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32), p1.mean()))

    p1 = matrix_pred_test_full[:, class_id].copy()
    p1[p1 > 0.5] = 1
    p1[p1 <= 0.5] = 0
    print('True labels in predictions for 0.5 default THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32),
                                                                                        p1.mean()))

    p1 = matrix_pred_test_full[:, class_id].copy()
    p1[p1 > best_thr] = 1
    p1[p1 <= best_thr] = 0
    print('True labels in predictions for optimal THR: {}. Fraction: {:.6f}'.format(p1.sum().astype(np.int32), p1.mean()))

    n_bins = 25

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    st = fig.suptitle('Class [ID: {}]: {} {}'.format(class_id, dcode[index_arr_backward[class_id]], index_arr_backward[class_id]), fontsize=14)
    # shift subplots down:
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    axs[0].hist(matrix_pred_valid[:, class_id], bins=n_bins, normed=True)
    axs[0].set_title('Valid')
    axs[1].hist(matrix_pred_test_full[:, class_id], bins=n_bins, normed=True)
    axs[1].set_title('Test full')
    axs[2].hist(matrix_pred_test[:, class_id], bins=n_bins, normed=True)
    axs[2].set_title('Test tuning')

    # plt.title('Class [ID: {}]: {} {}'.format(class_id, dcode[index_arr_backward[class_id]], index_arr_backward[class_id]))
    plt.show()
    plt.close()

    if 0:
        plt.plot(list(matrix_pred_valid[:, class_id]))
        plt.show()
        plt.close()

        plt.plot(list(matrix_pred_test[:, class_id]))
        plt.show()
        plt.close()


def get_score_for_thr(matrix_pred_valid, matrix_real_valid, preds, class_id, thr):
    preds2 = preds.copy()
    p1 = matrix_pred_valid[:, class_id].copy()
    p1[p1 > thr] = 1
    p1[p1 <= thr] = 0
    preds2[:, class_id] = p1

    score_valid = fbeta_score(matrix_real_valid, preds2, beta=2, average='samples')
    print("Optimal thr search: {:.2f} {:.6f}".format(thr, score_valid))
    return score_valid, thr


def get_tst_optimal_treshold_v1(valid_pred, best_thr, test_array, fraction_valid, fraction_train):
    # We assume that distribution in train and test are close to each other
    p = len(valid_pred[valid_pred > best_thr])
    p /= len(valid_pred)
    p *= len(test_array)
    p /= fraction_valid
    p *= fraction_train
    p = int(p)
    if p >= len(test_array):
        p = len(test_array) - 1
    test_array = np.sort(test_array)[::-1]
    optim_thr = test_array[p]
    return optim_thr


def tune_thresholds_validation(matrix_pred_valid, matrix_real_valid, matrix_pred_test, matrix_real_test, matrix_pred_test_full):
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    dcode, dreal = get_classes_dict()
    classes_distrib = get_classes_distribution_in_train()

    cache_path = OUTPUT_PATH + 'optimal_thresholds_cache.pklz'

    p = ThreadPool(7)

    # Find stat for validation
    if not os.path.isfile(cache_path):
        optimal_thr = []
        optimal_thr_test = []
        processed = []
        for class_id in range(0, matrix_pred_valid.shape[1]):
            optimal_thr.append(0.5)
            optimal_thr_test.append(0.5)
            processed.append(0)
    else:
        optimal_thr, optimal_thr_test, processed = load_from_file_fast(cache_path)

    preds_test = matrix_pred_test.copy()
    preds_test[preds_test > 0.5] = 1
    preds_test[preds_test <= 0.5] = 0
    score_test = fbeta_score(matrix_real_test, preds_test, beta=2, average='samples')
    print('Initial test score: {:.6f}'.format(score_test))

    for class_id in range(0, matrix_pred_valid.shape[1]):
        if processed[class_id]:
            continue
        if class_id in classes_distrib:
            if classes_distrib[class_id] < 0.05:
                continue
        else:
            continue
        fraction_train = classes_distrib[class_id]
        fraction_valid = (matrix_real_valid[:, class_id] > 0).astype(np.int32).sum() / matrix_real_valid.shape[0]
        print('Go for class: {} [Fraction train: {:.6f} Fraction valid: {:.6f}]'.format(class_id, fraction_train, fraction_valid))
        preds = matrix_pred_valid.copy()
        preds[preds > optimal_thr] = 1
        preds[preds <= optimal_thr] = 0

        check_thrs = [0.5, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        res = p.starmap(get_score_for_thr, [[matrix_pred_valid, matrix_real_valid, preds, class_id, thr] for thr in check_thrs])
        res = sorted(res, key=lambda tup: tup[0], reverse=True)
        best_score = res[0][0]
        best_thr = res[0][1]

        if 0:
            for thr in check_thrs:
                score_valid, thr_ret = get_score_for_thr(matrix_pred_valid, matrix_real_valid, preds, class_id, thr)
                print("Optimal thr search: {:.2f} {:.6f}".format(thr, score_valid))
                if score_valid > best_score:
                    best_score = score_valid
                    best_thr = thr
        print('Best validation score: {:.6f} for thr: {}'.format(best_score, best_thr))
        optimal_thr[class_id] = best_thr
        optimal_thr_test[class_id] = get_tst_optimal_treshold_v1(matrix_pred_valid[:, class_id].copy(), best_thr,
                                                                  matrix_pred_test_full[:, class_id].copy(),
                                                                  fraction_valid, fraction_train)
        processed[class_id] = 1
        save_in_file_fast((optimal_thr, optimal_thr_test, processed), cache_path)

        preds_test = matrix_pred_test.copy()
        preds_test[preds_test > optimal_thr_test] = 1
        preds_test[preds_test <= optimal_thr_test] = 0
        score_test = fbeta_score(matrix_real_test, preds_test, beta=2, average='samples')
        print('Current test score: {:.6f} Optimal test thr calculated as: {:.6f}'.format(score_test, optimal_thr_test[class_id]))


if __name__ == '__main__':
    start_time = time.time()
    # Experiment with optimal THR
    matrix_pred, matrix_real = get_tuning_matrix_from_cache_tst(OUTPUT_PATH + 'cache_resnet50_sh_336_test_v4/',
                                                     OUTPUT_PATH + 'cache_resnet50_sh_336_test_v4.pklz')
    start_point = 0.1
    end_point = 0.9
    min_number_of_entries = 1
    default_value = 0.9
    thr_arr = find_optimal_score(matrix_pred, matrix_real, start_point, end_point, min_number_of_entries, default_value)
    save_in_file(thr_arr, OUTPUT_PATH + 'thr_arr_resnet50_sh_336_sp_{}_ep_{}_min_{}_def_{}.pklz'.
                 format(start_point, end_point, min_number_of_entries, default_value))
    matrix_pred = create_submission(OUTPUT_PATH + 'cache_resnet50_sh_336_test/', thr_arr, SUBM_PATH + 'optim_thr_resnet50.csv')
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
Full score: 0.660907 [THR: 0.05]
Full score: 0.679815 [THR: 0.08]
Full score: 0.682401 [THR: 0.10] F2 test: 0.256851 [THR: 0.10]
Full score: 0.680763 [THR: 0.12]
Full score: 0.674153 [THR: 0.15]
Full score: 0.657085 [THR: 0.20] F2 test: 0.295527 [THR: 0.20]
Full score: 0.635852 [THR: 0.25] F2 test: 0.306785 [THR: 0.25]
Full score: 0.509631 [THR: 0.50] F2 test: 0.331256 [THR: 0.50]
Full score: 0.453765 [THR: 0.60] F2 test: 0.328287 [THR: 0.60]
Full score: 0.358289 [THR: 0.75] F2 test: 0.314678 [THR: 0.75]
Full score: 0.231647 [THR: 0.90] F2 test: 0.277767 [THR: 0.90]

Optimal THR:
[THR: 0.10 - 0.90, Min entries: 10] LS: 0.382048 LB: 0.374 
[THR: 0.05 - 0.95, Min entries: 5] LS: 0.416311 LB: 0.397
[THR: 0.01 - 0.99, Min entries: 3] LS: 0.447665 LB: 0.416
[THR: 0.001 - 0.999, Min entries: 3] LS: 0.451216 LB: 0.411

v2:
[THR: 0.01 - 0.99, Min entries: 3] LS: 0.452407 LB: 0.420 - default score: 0.5
[THR: 0.01 - 0.99, Min entries: 3] LS: 0.491802 LB: 0.452 - default score: 0.99999999
[THR: 0.01 - 0.99, Min entries: 3] LS: 0.506910 LB: 0.462 - default score: 0.99999999 - fix bug
[THR: 0.01 - 0.99, Min entries: 3] LS: 0.529965 LB: 0.474 - default score: 0.5 (for existed) and 0.99999999 (others)
[THR: 0.01 - 0.99, Min entries: 2] LS: 0.558764 LB: 0.481 - default score: 0.5 (for existed) and 0.99999999 (others)
[THR: 0.01 - 0.99, Min entries: 2] LS: 0.558764 LB: 0.480 - default score: 0.5 (for existed) and 0.99999999 (others) - fix for maximum
[THR: 0.01 - 0.99, Min entries: 1] LS: 0.603589 LB: 0.491 - default score: 0.5 (for existed) and 0.999999 (others)
[THR: 0.02 - 0.98, Min entries: 1] LS: 0.594130 LB: 0.485 - default score: 0.5 (for existed) and 0.999999 (others)
[THR: 0.01 - 0.99, Min entries: 1] LS: 0.595395 LB: 0.488 - default score: 0.5 (for existed) and 0.999999 (others) - low classes model
[THR: 0.01 - 0.99, Min entries: 1] LS: 0.619921 LB: 0.501 - default score: 0.5 (for existed) and 0.999999 (others) - better resnet
[THR: 0.01 - 0.99, Min entries: 1] LS: 0.608822 LB: 0.??? - default score: 0.5 (for existed) and 0.99 (others)
'''