# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a02_common_training_structures import *


CACHE_PATH_TEST = OUTPUT_PATH + 'cache_inception_resnet_v2_test_v2/'


def create_submission(folder_path, thr_arr, out_file):
    index_arr_forward, index_arr_backward = get_classes_to_index_dicts()
    out = open(out_file, 'w')
    out.write('image_id,labels\n')
    test_files = glob.glob(TEST_IMAGES_PATH + '*.jpg')
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


if __name__ == '__main__':
    start_time = time.time()
    start_point = 0.1
    end_point = 0.9
    min_number_of_entries = 3
    default_value = 0.9
    thr_arr = load_from_file(OUTPUT_PATH + 'thr_arr_inception_resnet_v2_sp_{}_ep_{}_min_{}_def_{}.pklz'.
                 format(start_point, end_point, min_number_of_entries, default_value))
    matrix_pred = create_submission(CACHE_PATH_TEST, thr_arr,
                                    SUBM_PATH + 'inception_resnet_v2.csv')
